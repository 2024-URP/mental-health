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

def asymmetric_loss(outputs, targets, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=False):
  lossfn = AsymmetricLossOptimized(gamma_neg, gamma_pos, clip, eps, disable_torch_grad_focal_loss)
  return lossfn(outputs, targets)
  
# 메모리 공간을 효울적으로 사용하기 위해 최적화된 loss함수
class AsymmetricLossOptimized(nn.Module):
    ''' Notice - optimized version, minimizes memory allocation and gpu uploading,
    favors inplace operations'''

    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=False):
        super(AsymmetricLossOptimized, self).__init__()

        self.gamma_neg = gamma_neg # negative 샘플 decay factor γ
        self.gamma_pos = gamma_pos # positive 샘플 decay factor γ
        self.clip = clip # asymmetric clipping의 margin
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss # Torch의 grad를 비활성화할지 여부
        self.eps = eps # Log의 진수가 0이되지 않게 해주는 epsilon

        # prevent memory allocation and gpu uploading every iteration, and encourages inplace operations
				# 메모리 확보를 위한 변수 초기화
        self.targets = self.anti_targets = self.xs_pos = self.xs_neg = self.asymmetric_w = self.loss = None

    def forward(self, output, target):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        self.targets = target # ground truth target
        self.anti_targets = 1 - target # 실제 target과 반대

        # Calculating Probabilities
        self.xs_pos = torch.sigmoid(output) # 모델의 출력을 sigmoid 함수로 확률값으로 변환 후, positive sample의 확률로 그대로 사용
        self.xs_neg = 1.0 - self.xs_pos # negative sample에 대해서는 1에서 뺀 값으로 수정: 모델이 잘 예측한 경우 1에 가까워짐

        # Asymmetric Clipping(shifting)
        if self.clip is not None and self.clip > 0:
            self.xs_neg.add_(self.clip).clamp_(max=1) # x_sigmoid가 self.clip 이하인 경우 1로 변환

        # Basic CE calculation, epsilon을 더해줌
        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps)) # positive sample에 대한 cross entropy 손실 계산
        self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps))) # negative sample에 대한 cross entropy 손실 계산

        # Asymmetric Focusing
        if (self.gamma_neg is not None) or (self.gamma_pos is not None):
            if self.disable_torch_grad_focal_loss: # gradient 계산 비활성화 설정이 되어있던 경우
                torch.set_grad_enabled(False) # gradient 계산 비활성화
            self.xs_pos = self.xs_pos * self.targets # ground truth 값을 곱하여 실제로 positive인 sample에 대한 확률만 남김
            self.xs_neg = self.xs_neg * self.anti_targets # 실제로 negative인 sample에 대한 1-p 확률만 남김
            self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg,
                                          self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets) # 각 확률에 곱해질 가중치 계산
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True) # gradient 계산 활성화
            self.loss *= self.asymmetric_w # 각 확률에 가중치를 곱해줌

        return -self.loss.sum(),-self.loss.sum()