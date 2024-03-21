from base import BaseModel
import warnings
warnings.filterwarnings("ignore")

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
        x = outputs.last_hidden_state[:, 0, :]   # [CLS] pooling
        # x = outputs.last_hidden_state.mean(1)  # [bs, seq_len, hidden_size] -> [bs, hidden_size]
        x2 = self.dropout(x)
        logits = self.clf(x2)
        return x, logits

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
