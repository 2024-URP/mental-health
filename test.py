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
from sklearn.metrics import classification_report
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
        split='test',
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
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)
    print(model.state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))
    
    all_losses = []
    all_targets = []
    all_probs = []
    all_full_loss = []

    with torch.no_grad():
        for i, (data, target, mask) in enumerate(tqdm(data_loader)):
            input_ids, attention_mask, token_type_ids = data['input_ids'], data['attention_mask'], data['token_type_ids']
            input_ids, attention_mask, token_type_ids = input_ids.to(device), attention_mask.to(device), token_type_ids.to(device)
            target, mask = target.to(device), mask.to(device)
            
            embeds, output = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            probs = torch.sigmoid(output)
                        
            # computing loss, metrics on test set
            full_loss, loss = loss_fn(output, target, mask)
            if contrastive : 
                c_loss = contrastive_loss(embeds, target, learning_temp=10)
                loss = loss + contrastive_gamma*c_loss

            batch_size = target.shape[0]
     
            all_full_loss.append(torch.mean(full_loss, axis=0))
            all_probs.append(probs)
            all_targets.append(target)
            all_losses.append(loss)
        
        avg_loss = torch.stack(all_losses).mean().detach().cpu().item()
        all_targets = np.concatenate([x.detach().cpu().numpy() for x in all_targets])
        all_probs = np.concatenate([x.detach().cpu().numpy() for x in all_probs])
        all_full_loss = np.mean([x.detach().cpu().numpy() for x in all_full_loss], axis=0)
        
        for i, metric in enumerate(metric_fns):
            total_metrics[i] += metric(all_targets, all_probs)

        # ### single
        # all_preds = np.where(all_probs<0.5, 0, 1)
        # report = classification_report(all_targets, all_preds)
        # ###

        ### multi
        all_preds = np.where(all_probs < 0.5, 0, 1)
        for i in range(all_targets.shape[1]):
            sel_indices = np.where(all_targets[:, i] != -1)
            report = classification_report(all_targets[:, i][sel_indices], all_preds[:, i][sel_indices])
            logger.info(f'======== {i} report==========')
            logger.info(report)
        logger.info(all_full_loss)
        ###
        
        out_dir = os.path.join(f'./saved/results/{EXP_NAME}', str_time)
        os.makedirs(out_dir, exist_ok=True)
        infer_output_file = os.path.join(out_dir, f"probs.npy")
        np.save(infer_output_file, all_probs)
    
    
    # ### single
    # logger.info(report)
    # ###
    
    ### multi
    log = {'loss':avg_loss}
    log.update({met.__name__: total_metrics[i].item() for i, met in enumerate(metric_fns)})
    logger.info(log)
    ###

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