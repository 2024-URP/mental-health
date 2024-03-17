import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker
from model.loss import contrastive_loss
from model.metric import print_classification_report

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
        
        # for contrastive loss
        self.contrastive, self.contrastive_gamma = contrastive
        
        # for metrics
        self.num_labels = config["arch"]["args"]["num_symps"]
        self.threshold = config["metrics"]["threshold"]
        self.metric_target = config["metrics"]['target']
                
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
        
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        best = self.model.train()
        self.train_metrics.reset()

        all_targets = np.array([])
        all_outputs = np.array([])

        for batch_idx, (data, target, mask) in enumerate(self.data_loader):
            input_ids, attention_mask, token_type_ids = data['input_ids'], data['attention_mask'], data['token_type_ids']
            input_ids, attention_mask, token_type_ids = input_ids.to(self.device), attention_mask.to(self.device), token_type_ids.to(self.device)
            target, mask = target.to(self.device), mask.to(self.device)
            
            self.optimizer.zero_grad()
            embeds, output = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)       
            _, loss = self.criterion(output, target, mask)

            # contrastive learning
            if self.contrastive : 
                c_loss = contrastive_loss(embeds, target, learning_temp=10)
                loss = loss + self.contrastive_gamma*c_loss
                
            loss.backward()
            self.optimizer.step()
            
            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update("loss", loss.item())
            
            output = torch.sigmoid(output)

            if batch_idx == 0 :
              all_targets = target.detach().cpu().numpy()
              all_outputs = output.detach().cpu().numpy()
            else :
              all_targets = np.concatenate((all_targets, target.detach().cpu().numpy()), axis=0)            
              all_outputs = np.concatenate((all_outputs, output.detach().cpu().numpy()), axis=0)
  
            
            if batch_idx % self.log_step == 0:
                print('Train Epoch : {} {}'.format(epoch, self._progress(batch_idx), loss.item()))
                
            if batch_idx == self.len_epoch:
                break

        all_targets = torch.from_numpy(all_targets)
        all_outputs = torch.from_numpy(all_outputs)

        for met in self.metric_ftns:
            self.train_metrics.update(met.__name__, met(self.threshold, self.num_labels, all_targets, all_outputs))
        
        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})
            log.update({"report" : self.report})

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
            
            all_targets = np.array([])
            all_outputs = np.array([])
        
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
              
                output = torch.sigmoid(output)
                if batch_idx == 0 :
                    all_targets = target.detach().cpu().numpy()
                    all_outputs = output.detach().cpu().numpy()
                else :
                    all_targets = np.concatenate((all_targets, target.detach().cpu().numpy()), axis=0)            
                    all_outputs = np.concatenate((all_outputs, output.detach().cpu().numpy()), axis=0)
           
                    
                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
        
        all_targets = torch.from_numpy(all_targets)
        all_outputs = torch.from_numpy(all_outputs)
            
        for met in self.metric_ftns:
            self.valid_metrics.update(met.__name__, met(self.threshold, self.num_labels, all_targets, all_outputs))

        self.report = print_classification_report(self.threshold, all_targets, all_outputs)
        
        # add histogram of model parameters to the tensorboard
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