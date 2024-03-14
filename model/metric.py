import torch
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
import numpy as np

def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)

def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)

def macro_auc(all_targets, all_outputs) :
    target_by_class = []
    output_by_class = []
    
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

def macro_f1(all_targets, all_outputs, threshold=0.5) :
    target_by_class = []
    output_by_class = []
    
    for i in range(all_targets.shape[1]):
        sel_indices = np.where(all_targets[:, i] != -1)
        target_by_class.append(all_targets[:, i][sel_indices])
        output_by_class.append(all_outputs[:, i][sel_indices])
    
    ret = []
    for target, output in zip(target_by_class, output_by_class):
        pred = (output > threshold).astype(float)
        ret.append(f1_score(target, pred))
        
    return np.mean(ret)

def macro_acc(all_targets, all_outputs, threshold=0.5) :
    target_by_class = []
    output_by_class = []
    
    for i in range(all_targets.shape[1]):
        sel_indices = np.where(all_targets[:, i] != -1)
        target_by_class.append(all_targets[:, i][sel_indices])
        output_by_class.append(all_outputs[:, i][sel_indices])
    
    ret = []
    for target, output in zip(target_by_class, output_by_class):
        pred = (output > threshold).astype(float)
        ret.append(np.mean(target == pred))
            
    return np.mean(ret)