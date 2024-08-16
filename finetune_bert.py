import torch as th
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F
from utils import *
import dgl
import torch.utils.data as Data
from ignite.engine import Events, create_supervised_evaluator, create_supervised_trainer, Engine
from torchmetrics.classification import MultilabelAccuracy, MultilabelPrecision, MultilabelRecall, MultilabelF1Score
from ignite.metrics import Accuracy, Loss, Metric
import numpy as np
import os
from datetime import datetime
from sklearn.metrics import accuracy_score
import argparse, shutil, logging
from torch.optim import lr_scheduler
from model import BertClassifier

parser = argparse.ArgumentParser()
parser.add_argument('--max_length', type=int, default=128, help='the input length for bert')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--nb_epochs', type=int, default=30)
parser.add_argument('--bert_lr', type=float, default=1e-4)
parser.add_argument('--dataset', default='psysym', choices=['20ng', 'R8', 'R52', 'ohsumed', 'mr','psysym'])
parser.add_argument('--bert_init', type=str, default='roberta-base',
                    choices=['roberta-base', 'roberta-large', 'bert-base-uncased', 'bert-large-uncased'])
parser.add_argument('--checkpoint_dir', default=None, help='checkpoint directory, [bert_init]_[dataset] if not specified')

args = parser.parse_args()

max_length = args.max_length
batch_size = args.batch_size
nb_epochs = args.nb_epochs
bert_lr = args.bert_lr
dataset = args.dataset
bert_init = args.bert_init
checkpoint_dir = args.checkpoint_dir
if checkpoint_dir is None:
    ckpt_dir = './checkpoint/{}_{}'.format(bert_init, dataset)
else:
    ckpt_dir = checkpoint_dir

os.makedirs(ckpt_dir, exist_ok=True)
shutil.copy(os.path.basename(__file__), ckpt_dir)

sh = logging.StreamHandler(sys.stdout)
sh.setFormatter(logging.Formatter('%(message)s'))
sh.setLevel(logging.INFO)
fh = logging.FileHandler(filename=os.path.join(ckpt_dir, 'training.log'), mode='w')
fh.setFormatter(logging.Formatter('%(message)s'))
fh.setLevel(logging.INFO)
logger = logging.getLogger('training logger')
logger.addHandler(sh)
logger.addHandler(fh)
logger.setLevel(logging.INFO)

cpu = th.device('cpu')
gpu = th.device('cuda:0')

logger.info('arguments:')
logger.info(str(args))
logger.info('checkpoints will be saved in {}'.format(ckpt_dir))

threshold = 0.5

# Data Preprocess
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size, test_size = load_corpus(dataset)
'''
y_train, y_val, y_test: n*c matrices 
train_mask, val_mask, test_mask: n-d bool array
train_size, test_size: unused
'''
#print(y_test)
# compute number of real train/val/test/word nodes and number of classes
nb_node = adj.shape[0]
nb_train, nb_val, nb_test = train_mask.sum(), val_mask.sum(), test_mask.sum()
nb_word = nb_node - nb_train - nb_val - nb_test
nb_class = y_train.shape[1]

# instantiate model according to class number
model = BertClassifier(pretrained_model=bert_init, nb_class=nb_class)

# transform one-hot label to class ID for pytorch computation
y = th.LongTensor((y_train + y_val +y_test))
#print(y)
label = {}
label['train'], label['val'], label['test'] = y[:nb_train], y[nb_train:nb_train+nb_val], y[-nb_test:]
#print(label)
# load documents and compute input encodings
corpus_file = './data/corpus/'+dataset+'_shuffle.txt'
with open(corpus_file, 'r') as f:
    text = f.read()
    text=text.replace('\\', '')
    text = text.split('\n')

def encode_input(text, tokenizer):
    input = tokenizer(text, max_length=max_length, truncation=True, padding=True, return_tensors='pt')
    return input.input_ids, input.attention_mask

input_ids, attention_mask = {}, {}

input_ids_, attention_mask_ = encode_input(text, model.tokenizer)

# create train/test/val datasets and dataloaders
input_ids['train'], input_ids['val'], input_ids['test'] =  input_ids_[:nb_train], input_ids_[nb_train:nb_train+nb_val], input_ids_[-nb_test:]
attention_mask['train'], attention_mask['val'], attention_mask['test'] =  attention_mask_[:nb_train], attention_mask_[nb_train:nb_train+nb_val], attention_mask_[-nb_test:]

datasets = {}
loader = {}
for split in ['train', 'val', 'test']:
    datasets[split] =  Data.TensorDataset(input_ids[split], attention_mask[split], label[split])
    loader[split] = Data.DataLoader(datasets[split], batch_size=batch_size, shuffle=True)

# Training

optimizer = th.optim.Adam(model.parameters(), lr=bert_lr)
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30], gamma=0.1)
'''
def masked_loss(logits, labels, masks=None, loss_weighting="mean", pos_weight=None, loss_type='bce', focal_gamma=2.):
    # treat unlabeled samples(-1) as implict negative (0.)
    labels2 = th.clamp_min(labels, 0.)
    #print(logits)
    losses = F.binary_cross_entropy_with_logits(logits, labels2, reduction='none', pos_weight=pos_weight)
    if loss_type == 'focal':
        # ref: https://discuss.pytorch.org/t/is-this-a-correct-implementation-for-focal-loss-in-pytorch/43327/14
        pt = torch.exp(-losses) # prevents nans when probability 0
        losses = (1-pt) ** focal_gamma * losses
    if masks is not None:
        masked_losses = th.masked_select(losses, masks)
        if loss_weighting == 'mean':
            return losses, masked_losses.mean()
        elif loss_weighting == 'geo_mean':
            # background ref: https://kexue.fm/archives/8870
            # numerical implementation: https://stackoverflow.com/questions/59722983/how-to-calculate-geometric-mean-in-a-differentiable-way
            masked_losses = masked_losses.mean(0)  # loss per task
            return losses, th.exp(torch.mean(torch.log(masked_losses)))
    else:
        if loss_weighting == 'mean':
            return losses, losses.mean()
        elif loss_weighting == 'geo_mean':
            losses = losses.mean(0)
            return losses, th.exp(torch.mean(torch.log(losses)))
'''

class MaskedLoss(th.nn.Module):
    def __init__(self, loss_weighting="mean", pos_weight=None, loss_type='bce', focal_gamma=2.):
        super(MaskedLoss, self).__init__()
        self.loss_weighting = loss_weighting
        self.pos_weight = pos_weight
        self.loss_type = loss_type
        self.focal_gamma = focal_gamma

    def forward(self, logits, labels, masks=None):
        # treat unlabeled samples(-1) as implict negative (0.)
        labels2 = th.clamp_min(labels, 0.)
        losses = F.binary_cross_entropy_with_logits(logits, labels2, reduction='none', pos_weight=self.pos_weight)
        
        if self.loss_type == 'focal':
            pt = th.exp(-losses) # prevents nans when probability 0
            losses = (1-pt) ** self.focal_gamma * losses
        
        if masks is not None:
            masked_losses = th.masked_select(losses, masks)
            if self.loss_weighting == 'mean':
                return masked_losses.mean()
            elif self.loss_weighting == 'geo_mean':
                masked_losses = masked_losses.mean(0)  # loss per task
                return th.exp(th.mean(th.log(masked_losses)))
        else:
            if self.loss_weighting == 'mean':
                return losses.mean()
            elif self.loss_weighting == 'geo_mean':
                losses = losses.mean(0)
                return th.exp(torch.mean(th.log(losses)))


def train_step(engine, batch):
    global model, optimizer
    model.train()
    model = model.to(gpu)
    optimizer.zero_grad()
    (input_ids, attention_mask, label) = [x.to(gpu) for x in batch]
    optimizer.zero_grad()
    y_pred = model(input_ids, attention_mask)
    y_true = label.type(th.long)
    #print(y_true)
    y_mask = (y_true>-1) # -1인 것들은 예측 대상에서 제외(False mask)
    #print(y_mask)
    #print(y_pred)
    #loss = F.cross_entropy(y_pred, y_true)
    loss_fn = MaskedLoss()
    loss = loss_fn(y_pred, y_true,masks=y_mask)
    #print(loss)
    loss.backward()
    optimizer.step()
    train_loss = loss.item()
    with th.no_grad():
        y_true = y_true.detach().cpu()
        y_pred = y_pred.detach().cpu()
        #print(y_true,y_pred)
        #y_pred = y_pred.argmax(axis=1).detach().cpu()
        #train_acc = accuracy_score(y_true, y_pred)
        metric_fn = MultilabelAccuracy(threshold=threshold, num_labels=nb_class)
        y_true[y_true==-1] = 0 # missing label
        train_acc = metric_fn(y_pred, y_true)
    return train_loss, train_acc


trainer = Engine(train_step)


def test_step(engine, batch):
    global model
    with th.no_grad():
        model.eval()
        model = model.to(gpu)
        (input_ids, attention_mask, label) = [x.to(gpu) for x in batch]
        optimizer.zero_grad()
        y_pred = model(input_ids, attention_mask)
        y_true = label
        print(y_pred, y_true)
        return y_pred, y_true

def thresholded_output_transform(output):
    y_pred, y = output
    y[y==-1] = 0
    result = th.zeros_like(y_pred, dtype=th.float16)
    result[y_pred >= threshold] = 1
    y_pred = result
    #print(y_pred, y)
    return y_pred, y

def thresholded_output_transform_ml(output):
    y_pred, y = output
    y_mask = (y>-1)
    '''
    y[y==-1] = 0
    result = th.zeros_like(y_pred, dtype=th.float16)
    result[y_pred >= threshold] = 1
    y_pred = result
    '''
    #print(y_pred, y)
    return y_pred, y, {'masks':y_mask}

evaluator = Engine(test_step)



metrics={
    'accuracy': Accuracy(is_multilabel=True, output_transform=thresholded_output_transform),
    'nll': Loss(MaskedLoss(), output_transform=thresholded_output_transform_ml),
}

for n, f in metrics.items():
    f.attach(evaluator, n)


@trainer.on(Events.EPOCH_COMPLETED)
def log_training_results(trainer):
    evaluator.run(loader['train'])
    metrics = evaluator.state.metrics
    train_acc, train_nll = metrics["accuracy"], metrics["nll"]
    evaluator.run(loader['val'])
    metrics = evaluator.state.metrics
    val_acc, val_nll = metrics["accuracy"], metrics["nll"]
    evaluator.run(loader['test'])
    metrics = evaluator.state.metrics
    test_acc, test_nll = metrics["accuracy"], metrics["nll"]
    logger.info(
        "\rEpoch: {}  Train acc: {:.4f} loss: {:.4f}  Val acc: {:.4f} loss: {:.4f}  Test acc: {:.4f} loss: {:.4f}"
        .format(trainer.state.epoch, train_acc, train_nll, val_acc, val_nll, test_acc, test_nll)
    )
    if val_acc > log_training_results.best_val_acc:
        logger.info("New checkpoint")
        th.save(
            {
                'bert_model': model.bert_model.state_dict(),
                'classifier': model.classifier.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': trainer.state.epoch,
            },
            os.path.join(
                ckpt_dir, 'checkpoint.pth'
            )
        )
        log_training_results.best_val_acc = val_acc
    scheduler.step()

        
log_training_results.best_val_acc = 0
trainer.run(loader['train'], max_epochs=nb_epochs)