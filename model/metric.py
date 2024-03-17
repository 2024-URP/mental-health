import torch
import numpy as np

from sklearn.metrics import roc_auc_score
from torchmetrics.classification import MultilabelAccuracy, MultilabelPrecision, MultilabelRecall, MultilabelF1Score
from sklearn.metrics import classification_report


def auc(threshold, num_labels, all_targets, all_outputs) :
    target_by_class = []
    output_by_class = []
    
    all_outputs = np.where(all_outputs<threshold, 0, 1) 
    
    for i in range(all_targets.shape[1]):
        sel_indices = np.where(all_targets[:, i] != -1)
        target_by_class.append(all_targets[:, i][sel_indices])
        output_by_class.append(all_outputs[:, i][sel_indices])
    
    ret = []
    for target, output in zip(target_by_class, output_by_class):
        try:
            ret.append(roc_auc_score(target, output))
        except:
            ret.append(0.5)
            
    return np.mean(ret)

def accuracy(threshold, num_labels, all_targets, all_outputs) :
    metric_fn = MultilabelAccuracy(threshold=threshold, num_labels=num_labels)
    all_targets[all_targets==-1] = 0 # missing label
    score = metric_fn(all_outputs, all_targets)
    return score 

def precision(threshold, num_labels, all_targets, all_outputs) :
    metric_fn = MultilabelPrecision(threshold=threshold, num_labels=num_labels)
    all_targets[all_targets==-1] = 0 # missing label
    score = metric_fn(all_outputs, all_targets)
    return score 

def recall(threshold, num_labels, all_targets, all_outputs) :
    metric_fn = MultilabelRecall(threshold=threshold, num_labels=num_labels)
    all_targets[all_targets==-1] = 0 # missing label
    score = metric_fn(all_outputs, all_targets)
    return score 

def f1_score(threshold, num_labels, all_targets, all_outputs) :
    metric_fn = MultilabelF1Score(threshold=threshold, num_labels=num_labels)
    all_targets[all_targets==-1] = 0 # missing label
    score = metric_fn(all_outputs, all_targets)
    return score 

def print_classification_report(threshold, all_targets, all_outputs) :
    all_targets[all_targets==-1] = 0 # missing label
    all_outputs = torch.where(all_outputs<threshold, 0, 1)
    report = classification_report(all_targets, all_outputs, output_dict=True)
    print(report)
    return report