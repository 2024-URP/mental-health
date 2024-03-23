from base import BaseModel
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from transformers import AutoModel


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
