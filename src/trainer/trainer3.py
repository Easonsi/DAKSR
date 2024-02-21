
import gc
import os
import time
import copy
import torch
import random
import datetime
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from numpy import random
from copy import deepcopy
import torch.optim as optim
from trainer.metrics import Metric
from config.configurator import configs
from models.bulid_model import build_model
from torch.utils.tensorboard import SummaryWriter
from .utils import DisabledSummaryWriter, log_exceptions
from trainer.logger import Logger

if 'tensorboard' in configs['train'] and configs['train']['tensorboard']:
    writer = SummaryWriter(log_dir='runs')
else:
    writer = DisabledSummaryWriter()

from trainer.trainer import Trainer
from trainer.tracker import DefaultMetricTracker, WAndBMetricTracker, set_metric_tracker
from models.kg.kgat import KGAT
from models.kg.kgin import KGIN
from models.kg_sequential.katrec import KATRec
from models.kg_sequential.ourkatcl import Ourkatcl



def format_eval_result(eval_result, k=configs['test']['k'], metrics=['recall'], part='test'):
    eval_result_dict = {}
    for _metric in metrics:
        # mrr, ndcg, recall
        for i in range(len(k)):
            eval_result_dict[f'{_metric}@{k[i]}'] = eval_result[_metric][i]
    result = {
        part: eval_result_dict
    }
    return result

class DakerTrainer(Trainer):
    def __init__(self, data_handler, logger:Logger):
        super().__init__(data_handler, logger)
        self.KGE = configs['model']['KGE']
        self.KGE_train_epoch = configs['model']['KGE_train_epoch']
        self.KGE_init = configs['model']['KGE_init']
        self.triplet_dataloader = data_handler.triplet_dataloader
        self.enable_wandb = configs['train']['enable_wandb']
    
    def create_optimizer(self, model):
        optim_config = configs['optimizer']
        if optim_config['name'] == 'adam':
            self.optimizer = optim.Adam(model.parameters(), lr=optim_config['lr'], weight_decay=optim_config['weight_decay'])
            self.kgtrans_optimizer = optim.Adam(model.parameters(), lr=optim_config['lr'], weight_decay=optim_config['weight_decay'])

    def _init_emb_transr(self, model:Ourkatcl):
        """ init emb with TransR """
        all_emb = model.all_embed.detach().cpu().numpy()
        item_emb = all_emb[: model.n_items]
        model.emb_layer.token_emb.weight.data[1:-1,:].copy_(torch.from_numpy(item_emb))

    @log_exceptions
    def train(self, model:Ourkatcl):
        is_distributed = 'distributed' in  configs['train'] and configs['train']['distributed'] and torch.cuda.device_count() > 1
        if is_distributed:
            model = nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
        
        self.create_optimizer(model)
        train_config = configs['train']

        if self.enable_wandb:
            import wandb
            wandb_config = {k : v for k, v in configs['model'].items()}
            wandb_config["dataset"] = configs['data']['name']
            wandb.init(
                project="SSL",
                config=wandb_config
            )
            wandb.watch(model)      # Hook into the torch model to collect gradients and the topology.
            wandb.define_metric("val_metric", summary="max")
            self.metric_tracker = WAndBMetricTracker()
        else:
            self.metric_tracker = DefaultMetricTracker()
        set_metric_tracker(self.metric_tracker)

        # 1] train KG with trans
        if self.KGE and self.KGE_init:
            for epoch_idx in range(train_config['epoch_trans']):
                self.train_epoch_trans(model, epoch_idx)

        # 2] train BERT
        now_patience = 0
        best_epoch = 0
        best_metric = -1e9
        best_state_dict = None
        for epoch_idx in range(train_config['epoch']):
            # train
            self.train_epoch(model, epoch_idx)
            # evaluate
            if epoch_idx % train_config['test_step'] == 0:
                eval_result_train = self.evaluate(model, epoch_idx, part='train')
                eval_result = self.evaluate(model, epoch_idx, part='valid')

                if eval_result[configs['test']['metrics'][0]][0] > best_metric:
                    now_patience = 0
                    best_epoch = epoch_idx
                    best_metric = eval_result[configs['test']['metrics'][0]][0]
                    if isinstance(model, nn.DataParallel):
                        best_state_dict = deepcopy(model.module.state_dict())
                    else:
                        best_state_dict = deepcopy(model.state_dict())
                else:
                    now_patience += 1

                # early stop
                if now_patience == configs['train']['patience']:
                    break

        self.logger.log("Best Epoch {}".format(best_epoch))
        model = build_model(self.data_handler).to(configs['device'])
        model.load_state_dict(best_state_dict)
        self.evaluate(model, part='test')
        self.test(model)
        self.save_model(model)

        if self.enable_wandb:
            wandb.finish()
        return model
    
    def train_epoch_trans(self, model:Ourkatcl, epoch_idx):
        loss_log_dict = {}
        """ train KG trans """
        triplet_dataloader = self.triplet_dataloader
        for _, tem in tqdm(enumerate(triplet_dataloader), desc='Training KG Trans', total=len(triplet_dataloader)):
            self.kgtrans_optimizer.zero_grad()
            batch_data = list(map(lambda x: x.long().to(configs['device']), tem))
            if isinstance(model, nn.DataParallel):
                kg_loss = model.module.cal_kg_loss(batch_data)
            else:
                kg_loss = model.cal_kg_loss(batch_data)
            kg_loss.backward()
            self.kgtrans_optimizer.step()
            
            if 'kg_loss' not in loss_log_dict:
                loss_log_dict['kg_loss'] = float(kg_loss) / len(triplet_dataloader)
            else:
                loss_log_dict['kg_loss'] += float(kg_loss) / len(triplet_dataloader)
        self.logger.log_loss(epoch_idx, loss_log_dict, mode='kg', save_to_log=True if configs['train']['log_loss'] else False)
        self.metric_tracker.log(loss_log_dict)
    
    def train_epoch(self, model:Ourkatcl, epoch_idx):
        model.train()
        train_dataloader = self.data_handler.train_dataloader
        train_dataloader.dataset.sample_negs()
        loss_log_dict = {}
        for batch_idx, tem in tqdm(enumerate(train_dataloader), desc='Training Recommender', total=len(train_dataloader)):
            self.optimizer.zero_grad()
            batch_data = list(map(lambda x: x.long().to(configs['device']), tem))
            if isinstance(model, nn.DataParallel):
                loss, loss_dict = model.module.cal_loss(batch_data, batch_idx=batch_idx)
            else:
                loss, loss_dict = model.cal_loss(batch_data, batch_idx=batch_idx)
            loss.backward()
            self.optimizer.step()

            for loss_name in loss_dict:
                _loss_val = float(loss_dict[loss_name]) / len(train_dataloader)
                if loss_name not in loss_log_dict:
                    loss_log_dict[loss_name] = _loss_val
                else:
                    loss_log_dict[loss_name] += _loss_val

        if self.KGE and self.KGE_train_epoch:
            triplet_dataloader = self.triplet_dataloader
            for _, tem in tqdm(enumerate(triplet_dataloader), desc='Training KG Trans', total=len(triplet_dataloader)):
                self.kgtrans_optimizer.zero_grad()
                batch_data = list(map(lambda x: x.long().to(configs['device']), tem))
                if isinstance(model, nn.DataParallel):
                    kg_loss = model.module.cal_kg_loss(batch_data)
                else:
                    kg_loss = model.cal_kg_loss(batch_data)
                kg_loss.backward()
                self.kgtrans_optimizer.step()

                if 'kg_loss' not in loss_log_dict:
                    loss_log_dict['kg_loss'] = float(kg_loss) / len(triplet_dataloader)
                else:
                    loss_log_dict['kg_loss'] += float(kg_loss) / len(triplet_dataloader)

        self.logger.log_loss(epoch_idx, loss_log_dict, save_to_log=configs['train']['log_loss'])
        self.metric_tracker.log(loss_log_dict)

    @log_exceptions
    def evaluate(self, model, epoch_idx=None, part='valid'):
        model.eval()
        if part == 'train':
            dataloader = self.data_handler.train_dataloader_raw
        elif part == 'valid':
            dataloader = self.data_handler.valid_dataloader if hasattr(self.data_handler, 'valid_dataloader') else self.data_handler.test_dataloader
        elif part == 'test':
            dataloader = self.data_handler.test_dataloader
        else:
            raise Exception("Invalid data part! Only 'train', 'valid' and 'test' are supported.")
        eval_result = self.metric.eval(model, dataloader)
        writer.add_scalar('HR/test', eval_result[configs['test']['metrics'][0]][0], epoch_idx)
        self.logger.log_eval(eval_result, configs['test']['k'], data_type='Test set', epoch_idx=epoch_idx)
        self.metric_tracker.log(format_eval_result(eval_result, part=part))
        return eval_result

