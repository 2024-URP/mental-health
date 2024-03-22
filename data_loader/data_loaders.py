from base import BaseDataLoader

import os
import re
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer,RobertaTokenizer
from torch.utils.data import Dataset
from collections import defaultdict

def my_collate_disease(data):
    labels = []
    processed_batch = defaultdict(list)
    for item, label in data:
        for k, v in item.items():
            processed_batch[k].append(v)
        labels.append(label)
    for k in ['input_ids', 'attention_mask', 'token_type_ids']:
        if k in processed_batch:  # roberta has no token_type_ids
            processed_batch[k] = torch.LongTensor(processed_batch[k])
    labels = torch.FloatTensor(labels)
    masks = torch.not_equal(labels, -1)
    return processed_batch, labels, masks

def preprocess_sent(sent):
    # remove hyperlink and preserve text mention
    sent = re.sub('\[(.*?)\]\(.*?\)', r'\1', sent)
    sent = sent.replace("[removed]", "")
    sent = sent.strip()
    return sent

def infer_preprocess(texts, tokenizer, max_len):
    texts = [preprocess_sent(text) for text in texts]
    tokenized = tokenizer(texts, truncation=True, padding='max_length', max_length=max_len)
    processed_batch = {}
    for k in ['input_ids', 'attention_mask', 'token_type_ids']:
        if k in tokenized:  #  has no token_type_ids
            processed_batch[k] = torch.LongTensor(tokenized[k])
    return processed_batch

class MultiDiseaseDataset(Dataset):
    def __init__(self, input_dir, tokenizer, max_len, uncertain, tokenizer_type, split="train"):
        assert split in {"train", "val", "test"}
        self.input_dir = input_dir
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.uncertain = uncertain
        self.data = []
        self.labels = []
        
        input_dir2 = os.path.join(input_dir, f"{split}.csv")
        df = pd.read_csv(input_dir2, index_col=None)
        
        if uncertain == 'exclude':
            self.symps = df.columns[5:].to_list() # multi
            # self.symps = df.columns[4:].to_list() # single
            self.symps.remove("uncertain")
        elif uncertain == 'include':
            self.symps = df.columns[5:].to_list() # multi
            # self.symps = df.columns[4:].to_list() # single
        else:
            self.symps = ['uncertain']

        self.is_control = []

        for rid, row in df.iterrows():
            sample = {}
            sample["text"] = row['sentence']
            tokenized = tokenizer(sample["text"], truncation=True, padding='max_length', max_length=max_len)
            if 'roberta-base' in tokenizer_type:
                tokenized = tokenizer.encode_plus(sample["text"], truncation=True, add_special_tokens=True,return_token_type_ids=True,padding='max_length', max_length=max_len)
            for k, v in tokenized.items():
                sample[k] = v
            self.data.append(sample)
            if uncertain == 'exclude':
                self.labels.append(row.values[5:-1]) # multi
                # self.labels.append(row.values[4:-1]) # single
            elif uncertain == 'include':
                self.labels.append(row.values[5:]) # multi
                # self.labels.append(row.values[4:]) # single
            else:
                self.labels.append(row.values[-1:])
            try :
              self.is_control.append(row['disease'] == 'control')
            except :
              self.is_control = []

        self.is_control = np.array(self.is_control).astype(int)
        self.label_counters = torch.zeros(len(self.symps), 2)
        for labels0 in self.labels:
            for class_id, label in enumerate(labels0):
                label = int(label)
                if label in {0, 1}:
                    self.label_counters[class_id, label] += 1

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        return self.data[index], self.labels[index]

class MultiDiseaseDataLoader(BaseDataLoader):
    def __init__(self, data_dir, tokenizer_type, batch_size, shuffle, split,
                 bal_sample, control_ratio, max_len, uncertain, num_workers, collate_fn=my_collate_disease):
        
        self.data_dir = data_dir
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_type, use_auth_token=True)
        self.dataset = MultiDiseaseDataset(self.data_dir, self.tokenizer, max_len, uncertain, tokenizer_type, split)
        
        super().__init__(self.dataset, batch_size, shuffle, split, bal_sample, control_ratio, num_workers, collate_fn)