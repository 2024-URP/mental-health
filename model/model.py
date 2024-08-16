from base import BaseModel
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from .torch_gcn import GCN
from .torch_gat import GAT


class BERTDiseaseClassifier(BaseModel):
    def __init__(self, model_type, num_symps) -> None:
        super().__init__()
        self.model_type = model_type
        self.num_symps = num_symps
        # multi-label binary classification
        self.encoder = AutoModel.from_pretrained(model_type, use_auth_token=True)
        self.dropout = nn.Dropout(self.encoder.config.hidden_dropout_prob)
        self.clf = nn.Linear(self.encoder.config.hidden_size, num_symps)
    
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, **kwargs):
        outputs = self.encoder(input_ids, attention_mask, token_type_ids)
        last_hidden_state = outputs.last_hidden_state   # [CLS] pooling
        # x = outputs.last_hidden_state[:, 0, :]   # [CLS] pooling
        # x = outputs.last_hidden_state.mean(1)  # [bs, seq_len, hidden_size] -> [bs, hidden_size]
        x = self.dropout(last_hidden_state[:, 0, :])
        logits = self.clf(x)
        return last_hidden_state, logits

class RoBERTaDiseaseClassifier(BaseModel):
    def __init__(self, num_symps):
        super().__init__()
        self.roberta = AutoModel.from_pretrained('roberta-base') # 사전학습된 BERT 모델 불러오기
        #self.l2 = nn.Dropout(0.3) # 정규화를 위한 dropout 레이어
        self.fc = nn.Linear(768,num_symps) # 선형레이어로 num_symps개 레이블에 대해 분류: roBERTa 모델의 출력은 768차원 벡터
        # 이진 분류를 위한 활성화함수

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, **kwargs):
        x, features = self.roberta(input_ids, attention_mask = attention_mask, token_type_ids = token_type_ids, return_dict=False)
        #output_2 = self.l2(output_1)
        output = self.fc(features) # 선형 레이어 학습
        return x, output
    
class CNNDiseaseClassifier(BaseModel):
    def __init__(self, model_type, num_symps) :
        super().__init__()
        self.model_type = model_type
        self.num_symps = num_symps
        self.encoder = AutoModel.from_pretrained(model_type, use_auth_token=True)
        
        self.embedding_dim = 768 # BERT
        self.filter_sizes = [3,4,5]
        self.n_filters = 128
        self.dropout_ratio = 0.2
         
        self.conv_layers = nn.ModuleList([
                                nn.Conv1d(in_channels=self.embedding_dim, out_channels=self.n_filters, kernel_size=filter_size)
                                for filter_size in self.filter_sizes
                            ])
        self.dropout = nn.Dropout(self.dropout_ratio)
        
        self.fc = torch.nn.Linear(in_features=len(self.filter_sizes) * self.n_filters, out_features=num_symps, bias=True)
    
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None):
        with torch.no_grad() :
            outputs = self.encoder(input_ids, attention_mask, token_type_ids)
            embedding = outputs.last_hidden_state
        
        x = embedding.permute(0,2,1) # change to (batch_size, embedding_dim, sequence_length)
        outs = [conv(x) for conv in self.conv_layers]
        outs = [nn.functional.max_pool1d(out, out.shape[2]).squeeze(2) for out in outs]
        out = self.dropout(torch.cat(outs, dim=1))
        out = self.fc(out)
        
        return None, out

class BertClassifier(torch.nn.Module):
    def __init__(self, pretrained_model='roberta_base', nb_class=20):
        super(BertClassifier, self).__init__()
        self.nb_class = nb_class
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.bert_model = AutoModel.from_pretrained(pretrained_model)
        self.feat_dim = list(self.bert_model.modules())[-2].out_features # BERT 모델의 출력 차원
        self.classifier = torch.nn.Linear(self.feat_dim, nb_class)

    def forward(self, input_ids, attention_mask):
        cls_feats = self.bert_model(input_ids, attention_mask)[0][:, 0]
        cls_logit = self.classifier(cls_feats)
        return cls_logit


class BertGCN(torch.nn.Module):
    def __init__(self, pretrained_model='roberta_base', nb_class=20, m=0.7, gcn_layers=2, n_hidden=200, dropout=0.5):
        super(BertGCN, self).__init__()
        self.m = m
        self.nb_class = nb_class
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.bert_model = AutoModel.from_pretrained(pretrained_model)
        self.feat_dim = list(self.bert_model.modules())[-2].out_features # bert_model의 forward에서 거치는 과정에서 마지막 선형레이어의 출력 차원(제일 마지막 과정은 tanh())
        self.classifier = torch.nn.Linear(self.feat_dim, nb_class) # 클래스 개수만큼 예측을 할 수 있도록 클래스 개수 크기만큼 출력
        self.gcn = GCN(
            in_feats=self.feat_dim,
            n_hidden=n_hidden,
            n_classes=nb_class,
            n_layers=gcn_layers-1,
            activation=F.elu,
            dropout=dropout
        ) # 사전에 정의해둔 GCN 모델: 최종 출력은 클래스 개수 크기

    def forward(self, g, idx):
        
        input_ids, attention_mask = g.ndata['input_ids'][idx], g.ndata['attention_mask'][idx] # 입력된 인덱스에 대해서만 그래프의 feature로 저장해둔 본문 토큰화결과와 attention mask을 꺼내옴
        if self.training: # 학습 중일 때 
            cls_feats = self.bert_model(input_ids, attention_mask)[0][:, 0] # BERT 모델을 통해 얻은 각 document마다 제일 첫번째 단어의 representation만 가져옴
            g.ndata['cls_feats'][idx] = cls_feats # 그래프의 cls_feat의 해당 인덱스 부분을 BERT의 출력값으로 대체함
        else:
            cls_feats = g.ndata['cls_feats'][idx] # 학습중이 아니면 그래프의 cls_feat의 해당 인덱스 부분을 가져옴  
        cls_logit = self.classifier(cls_feats) # 768차원 벡터를 분류기에 넣음
        cls_pred = torch.nn.Sigmoid()(cls_logit) # 어떤 클래스에 속할지 예측함
        gcn_logit = self.gcn(g.ndata['cls_feats'], g, g.edata['edge_weight'])[idx] # 그래프에 저장된 cls_feats을 각 document의 feature(입력)으로, 그래프, weight를 입력받아서 메세지 패싱을 하고 주어진 인덱스에 대해서만 뽑아냄
        gcn_pred = torch.nn.Sigmoid()(gcn_logit) # 어떤 클래스에 속할지 예측함
        pred = (gcn_pred+1e-10) * self.m + cls_pred * (1 - self.m) # self.m으로 두 예측의 비율 조정
        pred = torch.log(pred) # 로그 씌우기
        return pred # 예측값 출력
        '''
        gcn_logit = self.gcn(g.ndata['cls_feats'], g, g.edata['edge_weight'])[idx] # 그래프에 저장된 cls_feats을 각 document의 feature(입력)으로, 그래프, weight를 입력받아서 메세지 패싱을 하고 주어진 인덱스에 대해서만 뽑아냄
        gcn_pred = torch.nn.Softmax(dim=1)(gcn_logit)+1e-10
        return gcn_pred
        '''
    
class BertGAT(torch.nn.Module):
    def __init__(self, pretrained_model='roberta_base', nb_class=20, m=0.7, gcn_layers=2, heads=8, n_hidden=32, dropout=0.5):
        super(BertGAT, self).__init__()
        self.m = m
        self.nb_class = nb_class
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.bert_model = AutoModel.from_pretrained(pretrained_model)
        self.feat_dim = list(self.bert_model.modules())[-2].out_features
        self.classifier = torch.nn.Linear(self.feat_dim, nb_class)
        self.gcn = GAT(
                 num_layers=gcn_layers-1,
                 in_dim=self.feat_dim,
                 num_hidden=n_hidden,
                 num_classes=nb_class,
                 heads=[heads] * (gcn_layers-1) + [1],
                 activation=F.elu,
                 feat_drop=dropout,
                 attn_drop=dropout,
        )

    def forward(self, g, idx):
        input_ids, attention_mask = g.ndata['input_ids'][idx], g.ndata['attention_mask'][idx]
        if self.training:
            cls_feats = self.bert_model(input_ids, attention_mask)[0][:, 0]
            g.ndata['cls_feats'][idx] = cls_feats
        else:
            cls_feats = g.ndata['cls_feats'][idx]
        cls_logit = self.classifier(cls_feats)
        cls_pred = torch.nn.Softmax(dim=1)(cls_logit)
        gcn_logit = self.gcn(g.ndata['cls_feats'], g)[idx]
        gcn_pred = torch.nn.Softmax(dim=1)(gcn_logit)
        pred = (gcn_pred+1e-10) * self.m + cls_pred * (1 - self.m)
        pred = torch.log(pred)
        return pred