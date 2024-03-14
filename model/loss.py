import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

def nll_loss(output, target):
    return F.nll_loss(output, target)

def masked_loss(logits, labels, masks=None, loss_weighting="mean", pos_weight=None, loss_type='bce', focal_gamma=2.):
    # treat unlabeled samples(-1) as implict negative (0.)
    labels2 = torch.clamp_min(labels, 0.)
    losses = F.binary_cross_entropy_with_logits(logits, labels2, reduction='none', pos_weight=pos_weight)
    if loss_type == 'focal':
        # ref: https://discuss.pytorch.org/t/is-this-a-correct-implementation-for-focal-loss-in-pytorch/43327/14
        pt = torch.exp(-losses) # prevents nans when probability 0
        losses = (1-pt) ** focal_gamma * losses
    if masks is not None:
        masked_losses = torch.masked_select(losses, masks)
        if loss_weighting == 'mean':
            return losses, masked_losses.mean()
        elif loss_weighting == 'geo_mean':
            # background ref: https://kexue.fm/archives/8870
            # numerical implementation: https://stackoverflow.com/questions/59722983/how-to-calculate-geometric-mean-in-a-differentiable-way
            masked_losses = masked_losses.mean(0)  # loss per task
            return losses, torch.exp(torch.mean(torch.log(masked_losses)))
    else:
        if loss_weighting == 'mean':
            return losses, losses.mean()
        elif loss_weighting == 'geo_mean':
            losses = losses.mean(0)
            return losses, torch.exp(torch.mean(torch.log(losses)))

def distirbution_balanced_loss(
    logits, labels, masks=None, avg_factor=None, reduction='mean',
    reweight_func=None, loss_weight=1.0, 
    focal=dict(focal=True, alpha=0.5, gamma=2), 
    logit_reg=dict(), map_param=dict(),
                               ) :
    
    # FL
    # logits, labels, masks=None, avg_factor=None, reduction='mean',
    # reweight_func=None, loss_weight=1.0, 
    # focal=dict(focal=True, alpha=0.5, gamma=2), 
    # logit_reg=dict(), map_param=dict(),
    
    # R-FL
    # logits, labels, masks=None, avg_factor=None, reduction='mean',
    # reweight_func='rebalance', loss_weight=1.0, 
    # focal=dict(focal=True, alpha=0.5, gamma=2), 
    # logit_reg=dict(), map_param=dict(alpha=0.1, beta=10.0, gamma=0.55), 
    
    # NTR-FL
    # logits, labels, masks=None, avg_factor=None, reduction='mean',
    # reweight_func=None, loss_weight=1.0,
    # focal=dict(focal=True, alpha=0.5, gamma=2),
    # logit_reg=dict(init_bias=0.05, neg_scale=2.0), map_param=dict(),
    
    # DB-0FL
    # logits, labels, masks=None, avg_factor=None, reduction='mean',
    # reweight_func='rebalance', loss_weight=0.5,
    # focal=dict(focal=False, alpha=0.5, gamma=2),
    # logit_reg=dict(init_bias=0.05, neg_scale=2.0),
    # map_param=dict(alpha=0.1, beta=10.0, gamma=0.55), 
    
    # DB
    # logits, labels, masks=None, avg_factor=None, reduction='mean',
    # reweight_func='rebalance', loss_weight=1.0,
    # focal=dict(focal=True, alpha=0.5, gamma=2),
    # logit_reg=dict(init_bias=0.05, neg_scale=2.0),
    # map_param=dict(alpha=0.1, beta=10.0, gamma=0.55),
    
    class_freq = np.load('./model/dbloss/class_freq.npy')
    class_freq = torch.from_numpy(np.asarray(class_freq)).float().cuda()
    num_classes = class_freq.shape[0]
    train_num = 4238

    freq_inv = torch.ones(class_freq.shape).cuda() / class_freq
    propotion_inv = train_num / class_freq

    # regularization params
    neg_scale = logit_reg['neg_scale'] if 'neg_scale' in logit_reg else 1.0
    init_bias = logit_reg['init_bias'] if 'init_bias' in logit_reg else 0.0
    init_bias = - torch.log(propotion_inv - 1) * init_bias

    # treat unlabeled samples(-1) as implict negative (0.)
    labels = torch.clamp_min(labels, 0.)

    # reweight
    if reweight_func == 'rebalance' :
        # rebalance weight
        repeat_rate = torch.sum(labels.float() * freq_inv, dim=1, keepdim=True)
        pos_weight = freq_inv.clone().detach().unsqueeze(0) / repeat_rate
        map_alpha, map_beta, map_gamma = map_param['alpha'], map_param['beta'], map_param['gamma']
        weight = torch.sigmoid(map_beta * (pos_weight - map_gamma)) + map_alpha
    else :
        weight = None

    # logit regularization function 
    if 'init_bias' in logit_reg:
        logits += init_bias
    if 'neg_scale' in logit_reg:
        logits = logits * (1 - labels) * neg_scale  + logits * labels
        if weight is not None :
            weight = weight / neg_scale * (1 - labels) + weight * labels

    # focal
    if focal['focal'] :
        logpt = F.binary_cross_entropy_with_logits(logits, labels.float(), weight=None, reduction='none')
        pt = torch.exp(-logpt)
        
        if weight is not None :
            wtloss = F.binary_cross_entropy_with_logits(logits, labels.float(), weight=weight.float(), reduction='none')
        else : 
            wtloss = F.binary_cross_entropy_with_logits(logits, labels.float(), weight=None, reduction='none')

        focal_alpha, focal_gamma = focal['alpha'], focal['gamma']
        alpha_t = torch.where(labels==1, focal_alpha, 1-focal_alpha)
        
        loss = alpha_t * ((1-pt) ** focal_gamma) * wtloss

    else :
        loss = F.binary_cross_entropy_with_logits(logits, labels.float(), weight=weight.float(), reduction='none')
        
    if masks is not None:
        masked_losses = torch.masked_select(loss, masks) #2 : masked losses
    
    if reduction == 'mean' :
        masked_losses = masked_losses.mean()
    elif reduction == 'sum' :
        masked_losses = masked_losses.sum()

    masked_losses = loss_weight * masked_losses
    return loss, masked_losses

def contrastive_loss(embeds, labels, learning_temp=0.1) :    
    label_similarity = torch.mm(labels, labels.transpose(0,1))
    coefficient = label_similarity/torch.sum(label_similarity,1)

    distance = torch.cdist(embeds,embeds)
    distance2 = torch.exp(-distance/learning_temp)

    loss = -coefficient * torch.log(distance2/torch.sum(distance2, 1))   
    loss = torch.sum(loss)
    
    return loss