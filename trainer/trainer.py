import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker
from model.loss import contrastive_loss
from collections import defaultdict
from sklearn.metrics import classification_report

class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, contrastive, metric_ftns, optimizer, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.device = device
        self.data_loader = data_loader
        self.contrastive, self.contrastive_gamma = contrastive
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

        self.train_outputs = defaultdict(list)
        self.valid_outputs = defaultdict(list)
        
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        best = self.model.train()
        self.train_metrics.reset()

        for batch_idx, (data, target, mask) in enumerate(self.data_loader):
            input_ids, attention_mask, token_type_ids = data['input_ids'], data['attention_mask'], data['token_type_ids']
            input_ids, attention_mask, token_type_ids = input_ids.to(self.device), attention_mask.to(self.device), token_type_ids.to(self.device)
            target, mask = target.to(self.device), mask.to(self.device)
            
            self.optimizer.zero_grad()
            embeds, output = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            
            full_loss, loss = self.criterion(output, target, mask)

            # contrastive learning
            if self.contrastive : 
                c_loss = contrastive_loss(embeds, target, learning_temp=10)
                loss = loss + self.contrastive_gamma*c_loss
                
            loss.backward()
            self.optimizer.step()
            
            prob = torch.sigmoid(output)

            self.train_outputs['loss'].append(loss)
            self.train_outputs['targets'].append(target)
            self.train_outputs['outputs'].append(prob)
            self.train_outputs['full_loss'].append(torch.mean(full_loss, axis=0))  
            
            if batch_idx % self.log_step == 0: # single 실험에는 꺼도 됨
                print('Train Epoch : {} {}'.format(epoch, self._progress(batch_idx)))
                
            if batch_idx == self.len_epoch:
                break
        
        avg_loss = torch.stack(self.train_outputs['loss']).mean().detach().cpu().item()
        all_targets = np.concatenate([x.detach().cpu().numpy() for x in self.train_outputs['targets']])
        all_outputs = np.concatenate([x.detach().cpu().numpy() for x in self.train_outputs['outputs']])
        avg_full_loss = np.mean([x.detach().cpu().numpy() for x in self.train_outputs['full_loss']], axis=0)
    
        self.writer.set_step((epoch - 1))
        self.train_metrics.update('loss', avg_loss)
        
        for met in self.metric_ftns:
            self.train_metrics.update(met.__name__, met(all_targets, all_outputs))

        self.logger.debug('Train Epoch: {} Loss: {:.6f}'.format(epoch, avg_loss))
        
        log = self.train_metrics.result()
        log.update({'full_loss' : avg_full_loss})
        
        self.train_outputs.clear()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        #if self.lr_scheduler is not None:
        #    self.lr_scheduler.step()
        
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, (data, target, mask) in enumerate(self.valid_data_loader):
                input_ids, attention_mask, token_type_ids = data['input_ids'], data['attention_mask'], data['token_type_ids']
                input_ids, attention_mask, token_type_ids = input_ids.to(self.device), attention_mask.to(self.device), token_type_ids.to(self.device)
                target = target.to(self.device)
                mask = mask.to(self.device)
                
                embeds, output = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
                _, loss = self.criterion(output, target, mask)
                
                # contrastive learning
                if self.contrastive : 
                    c_loss = contrastive_loss(embeds, target, learning_temp=10)
                    loss = loss + self.contrastive_gamma*c_loss
              
                prob = torch.sigmoid(output)

                self.valid_outputs['loss'].append(loss)
                self.valid_outputs['targets'].append(target)
                self.valid_outputs['outputs'].append(prob)                
                    
            avg_loss = torch.stack(self.valid_outputs['loss']).mean().detach().cpu().item()
            all_targets = np.concatenate([x.detach().cpu().numpy() for x in self.valid_outputs['targets']])
            all_outputs = np.concatenate([x.detach().cpu().numpy() for x in self.valid_outputs['outputs']])

            self.writer.set_step((epoch - 1), 'valid')
            self.valid_metrics.update('loss', avg_loss)
            
            for met in self.metric_ftns:
                self.valid_metrics.update(met.__name__, met(all_targets, all_outputs))

            # ### single
            # all_outputs = np.where(all_outputs<0.5, 0, 1)
            # report = classification_report(all_targets, all_outputs)
            # self.logger.info(report)
            # ###
            
            ### multi 
            all_outputs = np.where(all_outputs<0.5, 0, 1) 
            for i in range(all_targets.shape[1]):
                sel_indices = np.where(all_targets[:, i] != -1)
                report = classification_report(all_targets[:, i][sel_indices], all_outputs[:, i][sel_indices])
                self.logger.info(f'======== {i} report==========')
                self.logger.info(report)
            ###

            self.valid_outputs.clear()

        # add histogram of model parameters to the tensorboard
        self.writer.add_text('classification report', report)
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)