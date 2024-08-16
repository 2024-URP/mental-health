import torch as th
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F
from utils import *
import dgl
import torch.utils.data as Data
from ignite.engine import Events, create_supervised_evaluator, create_supervised_trainer, Engine
from ignite.metrics import Accuracy, Loss
from torchmetrics.classification import MultilabelAccuracy, MultilabelPrecision, MultilabelRecall, MultilabelF1Score
from sklearn.metrics import accuracy_score
import numpy as np
import os
import shutil
import argparse
import sys
import logging
from datetime import datetime
from torch.optim import lr_scheduler
from model import BertGCN, BertGAT

parser = argparse.ArgumentParser()
parser.add_argument('--max_length', type=int, default=128, help='the input length for bert') # bert의 input 최대 길이(128)
parser.add_argument('--batch_size', type=int, default=64) # 배치 크기(64)
parser.add_argument('-m', '--m', type=float, default=0.7, help='the factor balancing BERT and GCN prediction') # BERT와 GCN의 예측을 조절하는 trade-off lambda(0.7)
parser.add_argument('--nb_epochs', type=int, default=30) # 에폭(50)
parser.add_argument('--bert_init', type=str, default='roberta-base',
                    choices=['roberta-base', 'roberta-large', 'bert-base-uncased', 'bert-large-uncased']) # BERT checkpoint
parser.add_argument('--pretrained_bert_ckpt', default=None) # 사전학습한 BERT의 checkpoint 불러오기(이걸해야 BertGCN의 성능이 유의미하다고 한 듯)
parser.add_argument('--dataset', default='psysym', choices=['20ng', 'R8', 'R52', 'ohsumed', 'mr','psysym']) # 데이터셋 선택(20ng)
parser.add_argument('--checkpoint_dir', default=None, help='checkpoint directory, [bert_init]_[gcn_model]_[dataset] if not specified') # BertGCN 모델을 저장할 위치. 이름은 기본 설정이 [bert_init]_[gcn_model]_[dataset]
parser.add_argument('--gcn_model', type=str, default='gcn', choices=['gcn', 'gat']) # GCN을 사용할지, GAT를 사용할지 선택(gcn)
parser.add_argument('--gcn_layers', type=int, default=2) # GCN 레이어 수(2)
parser.add_argument('--n_hidden', type=int, default=200, help='the dimension of gcn hidden layer, the dimension for gat is n_hidden * heads') # GNN이 연산할 때 feature의 차원(200)
parser.add_argument('--heads', type=int, default=8, help='the number of attentionn heads for gat') # GAT 사용할 때만 쓸듯?
parser.add_argument('--dropout', type=float, default=0.5) # dropout(0.5)
parser.add_argument('--gcn_lr', type=float, default=1e-3) # gcn의 learning rate(1e-3)
parser.add_argument('--bert_lr', type=float, default=1e-5) # bert의 learning rate(1e-5)

args = parser.parse_args()
max_length = args.max_length
batch_size = args.batch_size
m = args.m
nb_epochs = args.nb_epochs
bert_init = args.bert_init
pretrained_bert_ckpt = args.pretrained_bert_ckpt
dataset = args.dataset
checkpoint_dir = args.checkpoint_dir
gcn_model = args.gcn_model
gcn_layers = args.gcn_layers
n_hidden = args.n_hidden
heads = args.heads
dropout = args.dropout
gcn_lr = args.gcn_lr
bert_lr = args.bert_lr

if checkpoint_dir is None: # 경로 따로 지정안하면 경로와 파일명이 기본으로 결정됨
    ckpt_dir = './checkpoint/{}_{}_{}'.format(bert_init, gcn_model, dataset)
else: # 아니면 따로 저장
    ckpt_dir = checkpoint_dir
os.makedirs(ckpt_dir, exist_ok=True) # 경로 생성
shutil.copy(os.path.basename(__file__), ckpt_dir) 

# log 남기기
sh = logging.StreamHandler(sys.stdout)
sh.setFormatter(logging.Formatter('%(message)s'))
sh.setLevel(logging.INFO)
fh = logging.FileHandler(filename=os.path.join(ckpt_dir, 'training.log'), mode='w')
fh.setFormatter(logging.Formatter('%(message)s'))
fh.setLevel(logging.INFO)
logger = logging.getLogger('training logger')
logger.addHandler(sh)
logger.addHandler(fh)
logger.setLevel(logging.INFO)

# device 설정
cpu = th.device('cpu')
gpu = th.device('cuda:0')

logger.info('arguments:')
logger.info(str(args))
logger.info('checkpoints will be saved in {}'.format(ckpt_dir))
# Model


# Data Preprocess
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size, test_size = load_corpus(dataset)
'''
adj: n*n sparse adjacency matrix
y_train, y_val, y_test: n*c matrices 
train_mask, val_mask, test_mask: n-d bool array
'''

# compute number of real train/val/test/word nodes and number of classes
nb_node = features.shape[0] # 노드 개수
nb_train, nb_val, nb_test = train_mask.sum(), val_mask.sum(), test_mask.sum() # train, valid, test 데이터 수
nb_word = nb_node - nb_train - nb_val - nb_test # word 수
nb_class = y_train.shape[1] # 예측을 수행할 클래스 수

# instantiate model according to class number
if gcn_model == 'gcn': # gcn 모델을 입력받은 경우
    model = BertGCN(nb_class=nb_class, pretrained_model=bert_init, m=m, gcn_layers=gcn_layers,
                    n_hidden=n_hidden, dropout=dropout) # 예측을 수행할 클래스 수, 시작할 bert checkpoint, gcn 레이어 수, 히든 차원, dropout
else: # gat 모델을 입력받은 경우
    model = BertGAT(nb_class=nb_class, pretrained_model=bert_init, m=m, gcn_layers=gcn_layers,
                    heads=heads, n_hidden=n_hidden, dropout=dropout)


if pretrained_bert_ckpt is not None: # 이 task로 학습시켜둔 BERT 모델이 있는 경우
    ckpt = th.load(pretrained_bert_ckpt, map_location=gpu) # 학습시켜둔 모델 불러오기
    model.bert_model.load_state_dict(ckpt['bert_model']) # bert model의 가중치 가져오기
    model.classifier.load_state_dict(ckpt['classifier']) # bert model의 분류기 가중치 가져오기


# load documents and compute input encodings
corpse_file = './data/corpus/' + dataset +'_shuffle.txt' # 데이터 본문 전체
with open(corpse_file, 'r') as f:
    text = f.read()
    text = text.replace('\\', '')
    text = text.split('\n')

def encode_input(text, tokenizer):
    input = tokenizer(text, max_length=max_length, truncation=True, padding='max_length', return_tensors='pt')
#     print(input.keys())
    return input.input_ids, input.attention_mask # 토큰화


input_ids, attention_mask = encode_input(text, model.tokenizer) # 모델의 토크나이저로 토큰화
input_ids = th.cat([input_ids[:-nb_test], th.zeros((nb_word, max_length), dtype=th.long), input_ids[-nb_test:]]) # (train+word+test, max_length): 각 document의 토큰화 결과
attention_mask = th.cat([attention_mask[:-nb_test], th.zeros((nb_word, max_length), dtype=th.long), attention_mask[-nb_test:]]) # (train+word+test, max_length): 각 document의 마스킹 결과

# transform one-hot label to class ID for pytorch computation
y = y_train + y_test + y_val # 전체 라벨링 결과를 모두 더해줌(자리가 이미 정해져있으므로 순서 상관 없음)
#y_train = y_train.argmax(axis=1) # 학습 데이터의 라벨의 클래스 인덱스 배열(multi label classification에서 수정 필요)
#y = y.argmax(axis=1) # 전체 데이터의 라벨의 클래스 인덱스 배열(multi label classification에서 수정 필요)

# document mask used for update feature
doc_mask  = train_mask + val_mask + test_mask # word 제외 모든 document

# build DGL Graph
adj_norm = normalize_adj(adj + sp.eye(adj.shape[0])) # 인접행렬 자기자신에 1 더해준 후 정규화
g = dgl.from_scipy(adj_norm.astype('float32'), eweight_name='edge_weight') # adj_norm을 edge weight로하는 그래프 생성
g.ndata['input_ids'], g.ndata['attention_mask'] = input_ids, attention_mask # (train+word+test, max_length)
g.ndata['label'], g.ndata['train'], g.ndata['val'], g.ndata['test'] = \
    th.LongTensor(y), th.FloatTensor(train_mask), th.FloatTensor(val_mask), th.FloatTensor(test_mask) # (train+word+test, ) # g.ndata['label']: 모든 document의 클래스 인덱스
g.ndata['label_train'] = th.LongTensor(y_train) # (real_train_size, ) # g.ndata['label_train']: 학습 document의 클래스 인덱스
g.ndata['cls_feats'] = th.zeros((nb_node, model.feat_dim))

logger.info('graph information:')
logger.info(str(g))

# create index loader
train_idx = Data.TensorDataset(th.arange(0, nb_train, dtype=th.long))
val_idx = Data.TensorDataset(th.arange(nb_train, nb_train + nb_val, dtype=th.long))
test_idx = Data.TensorDataset(th.arange(nb_node-nb_test, nb_node, dtype=th.long))
doc_idx = Data.ConcatDataset([train_idx, val_idx, test_idx]) # 전체 인덱스

# 배치 크기만큼 인덱스 개수를 나눔
idx_loader_train = Data.DataLoader(train_idx, batch_size=batch_size, shuffle=True) 
idx_loader_val = Data.DataLoader(val_idx, batch_size=batch_size)
idx_loader_test = Data.DataLoader(test_idx, batch_size=batch_size)
idx_loader = Data.DataLoader(doc_idx, batch_size=batch_size, shuffle=True) # 전체 데이터셋에 대해 나눔

# Training
def update_feature(): # 한 에폭이 끝날 때 마다 수행됨
    global model, g, doc_mask
    # no gradient needed, uses a large batchsize to speed up the process
    dataloader = Data.DataLoader(
        Data.TensorDataset(g.ndata['input_ids'][doc_mask], g.ndata['attention_mask'][doc_mask]),
        batch_size=1024
    ) # 전체 데이터셋에 대해 1024 사이즈로 배치를 나눠 데이터 로드
    with th.no_grad():
        model = model.to(gpu)
        model.eval()
        cls_list = []
        for i, batch in enumerate(dataloader): # 모든 배치의 데이터셋에 대해 cls_feat을 업데이트
            input_ids, attention_mask = [x.to(gpu) for x in batch]
            output = model.bert_model(input_ids=input_ids, attention_mask=attention_mask)[0][:, 0]
            cls_list.append(output.cpu())
        cls_feat = th.cat(cls_list, axis=0)
    g = g.to(cpu)
    g.ndata['cls_feats'][doc_mask] = cls_feat
    return g # 그래프 출력


optimizer = th.optim.Adam([
        {'params': model.bert_model.parameters(), 'lr': bert_lr},
        {'params': model.classifier.parameters(), 'lr': bert_lr},
        {'params': model.gcn.parameters(), 'lr': gcn_lr},
    ], lr=1e-3
)
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30], gamma=0.1)

threshold = 0.5
class MaskedLoss(th.nn.Module):
    def __init__(self, loss_weighting="mean", pos_weight=None, loss_type='bce', focal_gamma=2.):
        super(MaskedLoss, self).__init__()
        self.loss_weighting = loss_weighting
        self.pos_weight = pos_weight
        self.loss_type = loss_type
        self.focal_gamma = focal_gamma

    def forward(self, logits, labels, masks=None):
        # treat unlabeled samples(-1) as implict negative (0.)
        labels2 = th.clamp_min(labels, 0.)
        losses = F.binary_cross_entropy_with_logits(logits, labels2, reduction='none', pos_weight=self.pos_weight)
        
        if self.loss_type == 'focal':
            pt = th.exp(-losses) # prevents nans when probability 0
            losses = (1-pt) ** self.focal_gamma * losses
        
        if masks is not None:
            masked_losses = th.masked_select(losses, masks)
            if self.loss_weighting == 'mean':
                return masked_losses.mean()
            elif self.loss_weighting == 'geo_mean':
                masked_losses = masked_losses.mean(0)  # loss per task
                return th.exp(th.mean(th.log(masked_losses)))
        else:
            if self.loss_weighting == 'mean':
                return losses.mean()
            elif self.loss_weighting == 'geo_mean':
                losses = losses.mean(0)
                return th.exp(torch.mean(th.log(losses)))

def train_step(engine, batch): # 매 배치(일부 document에 대한 feature에 관한 것)마다 반복하는 학습 과정
    global model, g, optimizer
    model.train()
    model = model.to(gpu)
    g = g.to(gpu)
    optimizer.zero_grad()
    (idx, ) = [x.to(gpu) for x in batch]
    optimizer.zero_grad()
    train_mask = g.ndata['train'][idx].type(th.BoolTensor)
    y_pred = model(g, idx)[train_mask] # 학습 데이터에 대해 BertGCN으로 예측한 idx의 값
    y_true = g.ndata['label_train'][idx][train_mask] # 학습 데이터 target으로 저장되어 있던 값들 가져오기
    #loss = F.nll_loss(y_pred, y_true) # 학습 데이터의 loss계산
    y_mask = (y_true>-1)
    loss_fn = MaskedLoss()
    loss = loss_fn(y_pred, y_true,masks=y_mask)
    loss.backward()
    optimizer.step() # 파라미터 업데이트
    g.ndata['cls_feats'].detach_()
    train_loss = loss.item()
    with th.no_grad():
        if train_mask.sum() > 0:
            y_true = y_true.detach().cpu()
            y_pred = y_pred.detach().cpu()
            #y_pred = y_pred.argmax(axis=1).detach().cpu()
            #train_acc = accuracy_score(y_true, y_pred)
            metric_fn = MultilabelAccuracy(threshold=threshold, num_labels=nb_class)
            y_true[y_true==-1] = 0 # missing label
            train_acc = metric_fn(y_pred, y_true)
        else:
            train_acc = 1
    return train_loss, train_acc


trainer = Engine(train_step) 


@trainer.on(Events.EPOCH_COMPLETED) # 한 에폭이 끝날 때마다 실행
def reset_graph(trainer):
    scheduler.step()
    update_feature() # g.ndata['cls_feats'] 중 주어진 인덱스에 대한 값들만 업데이트
    th.cuda.empty_cache()


def test_step(engine, batch):
    global model, g
    with th.no_grad():
        model.eval()
        model = model.to(gpu)
        g = g.to(gpu)
        (idx, ) = [x.to(gpu) for x in batch]
        y_pred = model(g, idx)
        y_true = g.ndata['label'][idx]
        return y_pred, y_true

def thresholded_output_transform(output):
    y_pred, y = output
    y[y==-1] = 0
    result = th.zeros_like(y_pred, dtype=th.float16)
    result[y_pred >= threshold] = 1
    y_pred = result
    #print(y_pred, y)
    return y_pred, y

def thresholded_output_transform_ml(output):
    y_pred, y = output
    y_mask = (y>-1)
    '''
    y[y==-1] = 0
    result = th.zeros_like(y_pred, dtype=th.float16)
    result[y_pred >= threshold] = 1
    y_pred = result
    '''
    #print(y_pred, y)
    return y_pred, y, {'masks':y_mask}

evaluator = Engine(test_step)

metrics={
    'accuracy': Accuracy(is_multilabel=True, output_transform=thresholded_output_transform),
    'nll': Loss(MaskedLoss(), output_transform=thresholded_output_transform_ml),
}

for n, f in metrics.items():
    f.attach(evaluator, n)


@trainer.on(Events.EPOCH_COMPLETED)
def log_training_results(trainer):
    evaluator.run(idx_loader_train)
    metrics = evaluator.state.metrics
    train_acc, train_nll = metrics["accuracy"], metrics["nll"]
    evaluator.run(idx_loader_val)
    metrics = evaluator.state.metrics
    val_acc, val_nll = metrics["accuracy"], metrics["nll"]
    evaluator.run(idx_loader_test)
    metrics = evaluator.state.metrics
    test_acc, test_nll = metrics["accuracy"], metrics["nll"]
    logger.info(
        "Epoch: {} Train acc: {:.4f} loss: {:.4f}  Val acc: {:.4f} loss: {:.4f}  Test acc: {:.4f} loss: {:.4f}"
        .format(trainer.state.epoch, train_acc, train_nll, val_acc, val_nll, test_acc, test_nll)
    )
    if val_acc > log_training_results.best_val_acc:
        logger.info("New checkpoint")
        th.save(
            {
                'bert_model': model.bert_model.state_dict(),
                'classifier': model.classifier.state_dict(),
                'gcn': model.gcn.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': trainer.state.epoch,
            },
            os.path.join(
                ckpt_dir, 'checkpoint.pth'
            )
        )
        log_training_results.best_val_acc = val_acc


log_training_results.best_val_acc = 0
g = update_feature()
trainer.run(idx_loader, max_epochs=nb_epochs)