import argparse
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from model.loss import contrastive_loss
from parse_config import ConfigParser
import numpy as np
import pandas as pd
import os
import datetime as dt
from huggingface_hub import login
from model.metric import print_classification_report

import json

# hugging face login
with open('./secret.json') as f :
    secret = json.loads(f.read())

TOKEN = secret['HUGGINGFACE_TOKEN']
login(token=TOKEN)

# token paralleism
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main(config):
    logger = config.get_logger('test')
    
    x = dt.datetime.now()
    str_time = x.strftime('%m%d_%H%M%S')

    EXP_NAME = config['name']

    # setup data_loader instances
    data_loader = getattr(module_data, config['data_loader']['type'])(
        config['data_loader']['args']['data_dir'],
        config['data_loader']['args']['tokenizer_type'],
        config['data_loader']['args']['batch_size'],
        shuffle = False,
        split='test', # or 'val'
        bal_sample=False,
        control_ratio=0.75,
        max_len=64,
        uncertain='exclude',
        num_workers=2
    )

    # build model architecture
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss']['name'])
    contrastive = config['loss']['contrastive']
    contrastive_gamma = config['loss']['contrastive_gamma']
    metric_fns = [getattr(module_metric, met) for met in config['metrics']['target']]

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)
    print(model.state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))
    
    all_targets = np.array([])
    all_outputs = np.array([])

    with torch.no_grad():
        for i, (data, target, mask) in enumerate(tqdm(data_loader)):
            input_ids, attention_mask, token_type_ids = data['input_ids'], data['attention_mask'], data['token_type_ids']
            input_ids, attention_mask, token_type_ids = input_ids.to(device), attention_mask.to(device), token_type_ids.to(device)
            target, mask = target.to(device), mask.to(device)
            
            last_hidden_state, output = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            _, loss = loss_fn(output, target, mask)
            
            # contrastive learning
            if contrastive : 
                c_loss = contrastive_loss(last_hidden_state[:,0,:], target, learning_temp=10)
                loss = loss + contrastive_gamma*c_loss
          
            output = torch.sigmoid(output)

            if i == 0 :
                all_targets = target.detach().cpu().numpy()
                all_outputs = output.detach().cpu().numpy()
            else :
                all_targets = np.concatenate((all_targets, target.detach().cpu().numpy()), axis=0)            
                all_outputs = np.concatenate((all_outputs, output.detach().cpu().numpy()), axis=0)
        
        all_targets = torch.from_numpy(all_targets)
        all_outputs = torch.from_numpy(all_outputs)

        for i, metric in enumerate(metric_fns):
          total_metrics[i] += metric(0.5, 38, all_targets, all_outputs) # threshold=0.5, num_classes(symps)=38
        
        report = print_classification_report(0.5, all_targets, all_outputs)

        logger.info({
            met.__name__: total_metrics[i].item() for i, met in enumerate(metric_fns)
        })
        logger.info({'report' : report})

        # saving probability file
        out_dir = os.path.join(f'./saved/results/{EXP_NAME}', str_time)
        os.makedirs(out_dir, exist_ok=True)
        infer_output_file = os.path.join(out_dir, f"probs.npy")
        np.save(infer_output_file, all_outputs)
    

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)