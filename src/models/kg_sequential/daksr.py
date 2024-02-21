import os
import pickle
import torch
from torch import nn
import torch.nn.functional as F
from config.configurator import configs
import random
from models.loss_utils import cal_bpr_loss
from models.base_model import BaseModel
from models.model_utils import TransformerLayer, TransformerEmbedding
from models.model_utils2 import GATransformerLayer, GASimpleTransformerLayer, HighwayGateLayer, GATransformerLayersV2
from data_utils.data_handler_kg_sequential import DataHandlerKGSequential
from models.kg_sequential.util import build_dist_mat_mp
import math
import numpy as np
from line_profiler import LineProfiler, profile
lp = LineProfiler()
import pandas as pd
import wandb

def _L2_loss_mean(x):
    return torch.mean(torch.sum(torch.pow(x, 2), dim=1, keepdim=False) / 2.)

class DAKSR(BaseModel):
    def __init__(self, data_handler: DataHandlerKGSequential):
        super(DAKSR, self).__init__(data_handler)
        self.item_num = configs['data']['item_num']
        self.emb_size = configs['model']['embedding_size']
        self.max_len = configs['model']['max_seq_len']
        self.KCL = configs['model']['KCL']
        self.KGE = configs['model']['KGE']
        self.DA = configs['model']['DA']
        
        self.mask_token = self.item_num + 1
        self.entity_num = configs['data']['entity_num']
        self.n_layers = configs['model']['n_layers']
        self.n_heads = configs['model']['n_heads']
        self.emb_size = configs['model']['embedding_size']
        self.max_depth = configs['model']['max_depth']
        self.th_dist = configs['model']['th_dist']
        self.ga_idx = configs['model']['ga_idx']
        self.inner_size = 4 * self.emb_size
        self.dropout_rate = configs['model']['dropout_rate']

        self.batch_size = configs['train']['batch_size']

        self.filter = configs['data']['filter']
        self.filter_threshold = configs['data']['filter_threshold']
        self.n_users = configs['data']['user_num']
        self.n_items = configs['data']['item_num']
        self.n_entities = configs['data']['entity_num']
        self.n_nodes = configs['data']['node_num']
        self.n_relations = configs['data']['relation_num']
        self.relation_mat = nn.Parameter(nn.init.normal_(torch.empty(self.n_relations + 2, 1), std=0.1))        # 最后一个是全图 (不区分relation的距离矩阵分数)
        self.entity_embed = nn.Parameter(nn.init.normal_(torch.empty(self.n_entities + 1, self.emb_size), std=0.1))
        self.relation_embed = nn.Parameter(nn.init.normal_(torch.empty(self.n_relations + 1, self.emb_size), std=0.1))


        self.kg_edges = data_handler.kg_edges
        self.kg_edges_raw = data_handler.kg_edges_raw

        self.DA_att_mode = configs['model']['DA_att_mode']
        self.DA_M_mode = configs['model']['DA_M_mode']
        self.ppr_temperature = configs['model']['ppr_temperature']
        self.ppr_mask_value = configs['model']['ppr_mask_value']
        self.ppr_rescale = configs['model']['ppr_rescale']
        
        self.kg_ppr_mat_raw = self._build_ppr_mat(self.kg_edges_raw, self.item_num)
        self.kg_ppr_mat2 = self._trans_ppr_score_v3(self.kg_ppr_mat_raw, mask_value=self.ppr_mask_value, temperature=self.ppr_temperature)
        self.kg_ppr_mat = self.kg_ppr_mat2.clone().detach().to(configs['device'])
        

        self.kg_dist_mat_raw = self._build_dist_mat(self.kg_edges, self.entity_num, self.item_num, max_depth=self.max_depth, relational=False)
        self.kg_dist_mat2 = self._trans_dist_score(self.kg_dist_mat_raw, ga_idx=self.ga_idx)
        self.kg_dist_mat = torch.tensor(self.kg_dist_mat2, dtype=torch.float32, requires_grad=False).to(configs['device']) # 为了节约显存, 采用 int8 类型
        
        # self.kg_dist_relational_mat_raw = self._build_dist_mat(self.kg_edges, self.entity_num, self.item_num, max_depth=self.max_depth, relational=True)
        # self.kg_dist_relational_mat2 = self._trans_multi_dist_score(self.kg_dist_relational_mat_raw, ga_idx=self.ga_idx)
        # self.kg_dist_relational_mat = torch.tensor(self.kg_dist_relational_mat2, dtype=torch.float32, requires_grad=False).to(configs['device']) # 为了节约显存, 采用 int8 类型

        self.kg_fake_mat = self._build_fake_mat(self.item_num).to(configs['device'])

        self.emb_layer = TransformerEmbedding(self.item_num + 2, self.emb_size, self.max_len)
        if self.DA:
            self.transformer_layers = GATransformerLayersV2(
                self.emb_size, self.n_heads, self.emb_size * 4, 
                dropout_rate=self.dropout_rate, n_layers=self.n_layers, mode=self.DA_att_mode)
        else:
            raise Exception("DA must be True! Please set `DA_M_mode=none` to disable DA!")
        
        if self.KGE:
            self.KGE_itemset = configs['model']['KGE_itemset']
            self.KGE_gate = HighwayGateLayer(in_out_size=self.emb_size)
            self.itemset_trans_mat = self._build_itemset_trans_mat(self.kg_dist_mat, self.th_dist)

        if self.KCL:
            self.KCL_kgaug = configs['model']['KCL_kgaug']
            self.lmd = configs['model']['lmd']
            self.tau = configs['model']['tau']
            self.crate = configs['model']['crate']
            self.c_th = configs['model']['c_th']
        
        self.loss_func = nn.CrossEntropyLoss()

        self.mask_default = self.mask_correlated_samples(
            batch_size=self.batch_size)
        self.cl_loss_func = nn.CrossEntropyLoss()

        # parameters initialization
        self.apply(self._init_weights)

    def _build_itemset_trans_mat(self, kg_dist_mat, th_dist=3):
        kg_dist_mat[-1][-1] = 1.
        dist_mat = (th_dist - kg_dist_mat).float()
        dist_mat[dist_mat < 0] = -torch.inf
        weights = torch.softmax(dist_mat.float(), dim=1)
        return weights

    def _build_dist_mat(self, kg_edges, num_entities, num_items, max_depth=5, default_value=100, relational=False):
        cache_fn = f"./datasets/kg_sequential/{configs['data']['name']}/kg_dist_mat"
        if relational:
            cache_fn += "_relational"
        if self.filter:
            cache_fn += f"_filter{self.filter_threshold}"
        cache_fn += ".pkl"
        if os.path.exists(cache_fn) and ('use_cache' not in configs['data'] or configs['data']['use_cache']):
            print(f"loading kg_dist_mat from cache {cache_fn}!!")
            with open(cache_fn, 'rb') as f:
                return pickle.load(f)
        
        print(f"building dist matrix...")
        dist_sp = build_dist_mat_mp(kg_edges, num_entities, num_items, max_depth=max_depth, default_value=default_value, workers=32, relational=relational)
        
        with open(cache_fn, 'wb') as f:
            pickle.dump(dist_sp, f)
        return dist_sp

    def _build_ppr_mat(self, kg_edges, n_items):
        cache_fn = f"./datasets/kg_sequential/{configs['data']['name']}/kg_ppr_mat.pkl"
        if os.path.exists(cache_fn):
            print(f"loading kg_dist_mat from cache {cache_fn}!!")
            # return torch.load(cache_fn)
            with open(cache_fn, 'rb') as f:
                return pickle.load(f)
        from torch_ppr import personalized_page_rank
        kg = pd.DataFrame(kg_edges, columns=['h', 'r', 't'])
        edge_index = torch.tensor(kg[['h', 't']].values.T, dtype=torch.long)
        kg_ppr_mat = torch.zeros((n_items+2, n_items+2))
        kg_ppr_mat[1:n_items+1,1:n_items+1] = personalized_page_rank(
            edge_index=edge_index, indices=list(range(1,n_items+1)))[:, 1:n_items+1]
        with open(cache_fn, 'wb') as f:
            pickle.dump(kg_ppr_mat, f)
        return kg_ppr_mat
    
    def _build_fake_mat(self, n_items):
        kg_fake_mat = torch.zeros((n_items+2, n_items+2))
        return kg_fake_mat

    def _trans_ppr_score_v3(self, x, mask_value=0, temperature=1, eps=1e-10):
        x[x==0] = mask_value
        x = x - x.min() + eps
        x = torch.log(x)
        a = torch.softmax(x / temperature, dim=1)
        return a

    def _trans_dist_score(self, x, ga_idx=1):
        # trans function: gamask = (1+dist_mat)^5
        a = 1 / (x ** ga_idx)
        return a
    def _trans_multi_dist_score(self, x, ga_idx=1):
        for i in range(x.shape[0]):
            a = 1 / (x[i] ** ga_idx)
            x[i] = a
        return x
    
    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def _cl4srec_aug(self, batch_seqs, c_th=5):
        from models.kg_sequential.util_cl import item_kg_crop, item_crop, item_mask, item_reorder, item_kg_reorder
        seqs = batch_seqs.tolist()
        lengths = batch_seqs.count_nonzero(dim=1).tolist()

        aug_seq1, aug_seq2 = [], []
        for seq, length in zip(seqs, lengths):
            if length <= 1: 
                aug_seq1.append(seq)
                aug_seq2.append(seq)
                continue
            seq = np.asarray(seq.copy(), dtype=np.int64)
            for idx, aug_seq in zip((0,1), (aug_seq1, aug_seq2)):
                if not self.KCL_kgaug:
                    switch = random.choice(range(3))
                    if switch == 0:
                        _s, _l = item_crop(seq, length)
                    elif switch == 1:
                        _s, _l = item_mask(seq, length, mask_token=self.mask_token)
                    elif switch == 2:
                        _s, _l = item_reorder(seq, length)
                    else: raise ValueError(f"switch {switch} not supported!")
                else:
                    switch = random.choice(range(3))
                    if switch == 0:
                        _s, _l = item_kg_crop(seq, length, dist_mat=self.kg_dist_mat_raw, crate=self.crate, c_th=self.c_th)
                    elif switch == 1:
                        _s, _l = item_mask(seq, length, mask_token=self.mask_token)
                    elif switch == 2:
                         _s, _l = item_kg_reorder(seq, length, dist_mat=self.kg_dist_mat_raw, crate=self.crate, c_th=self.c_th)
                    else: raise ValueError(f"switch {switch} not supported!")
                aug_seq.append(_s if _l > 0 else seq.tolist())

        aug_seq1 = torch.tensor(
            aug_seq1, dtype=torch.long, device=batch_seqs.device)
        aug_seq2 = torch.tensor(
            aug_seq2, dtype=torch.long, device=batch_seqs.device)
        return aug_seq1, aug_seq2

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def info_nce(self, z_i, z_j, temp, batch_size):
        N = 2 * batch_size

        z = torch.cat((z_i, z_j), dim=0)

        sim = torch.mm(z, z.T) / temp

        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        if batch_size != self.batch_size:
            mask = self.mask_correlated_samples(batch_size)
        else:
            mask = self.mask_default
        negative_samples = sim[mask].reshape(N, -1)

        labels = torch.zeros(N).long().to(configs['device'])
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        info_nce_loss = self.cl_loss_func(logits, labels)
        return info_nce_loss

    def _get_seq_mat(self, batch_seqs, kg_mat, mask_val=0):
        """ 对于每一个序列, 根据kg_dist_mat生成序列上的距离矩阵 """
        B,T = batch_seqs.size()
        idx_c = batch_seqs.unsqueeze(2).repeat(1,1,T)
        idx_r = batch_seqs.unsqueeze(1).repeat(1,T,1)
        dist_mat = kg_mat[idx_c, idx_r]
        # val 0 -> mask_val
        dist_mat[dist_mat == 0] = mask_val
        return dist_mat

    def forward_kgin(self):
        entity_emb = self.all_embed
        self.ent_emb_kgin = self.kgin(entity_emb, self.edge_index, self.edge_type, self.relation_embed)

    def get_itemset_emb(self, batch_seqs):
        item_emb = self.emb_layer.token_emb.weight
        itemset_emb = self.itemset_trans_mat @ item_emb
        return itemset_emb[batch_seqs]

    @profile
    def forward(self, batch_seqs, output_attentions=False):
        x = self.emb_layer(batch_seqs)
        
        if self.KGE and self.KGE_itemset:
            x_itemset = self.get_itemset_emb(batch_seqs)
            x = self.KGE_gate(x, x_itemset)
        mask = (batch_seqs > 0).unsqueeze(1).repeat(1, batch_seqs.size(1), 1).unsqueeze(1)
        
        if self.DA:
            if self.DA_M_mode == 'dist':
                dist_mat = self._get_seq_mat(batch_seqs, kg_mat=self.kg_dist_mat, mask_val=0)
                gamask = dist_mat
            elif self.DA_M_mode == 'relational_dist':
                gm = []
                for kg_dist_mat in self.kg_dist_relational_mat:
                    dist_mat = self._get_seq_mat(batch_seqs, kg_mat=kg_dist_mat, mask_val=0)
                    gm.append(dist_mat)
                gm.append(self._get_seq_mat(batch_seqs, kg_mat=self.kg_dist_mat, mask_val=0))
                gm = torch.stack(gm, dim=1).to(configs['device'])     # [bs, n_rel+1, T, T]
                # use self.relation_mat as attention
                gamask = (gm * self.relation_mat.reshape((1,-1,1,1))).sum(dim=1)    # [bs, T, T]
                # # set elements in eye to 0.
                gamask[:, range(gamask.shape[1]), range(gamask.shape[1])] = 0.
            elif self.DA_M_mode == 'ppr':
                ppr_mat = self._get_seq_mat(batch_seqs, kg_mat=self.kg_ppr_mat, mask_val=0)
                ppr_mat = ppr_mat / ppr_mat.sum(dim=-1).unsqueeze(-1) * self.ppr_rescale
                gamask = ppr_mat
            elif self.DA_M_mode == 'mix':
                dist_mat = self._get_seq_mat(batch_seqs, kg_mat=self.kg_dist_mat, mask_val=0)
                ppr_mat = self._get_seq_mat(batch_seqs, kg_mat=self.kg_ppr_mat, mask_val=0)
                gamask = dist_mat + ppr_mat
            elif self.DA_M_mode == 'none':
                gamask = None
            elif self.DA_M_mode == 'fake':
                gamask = self._get_seq_mat(batch_seqs, kg_mat=self.kg_fake_mat, mask_val=0)
            else:
                raise Exception(f"ppr_mode {self.ppr_mode} not supported!")
            if gamask is not None:
                gamask.unsqueeze_(1)
            x = self.transformer_layers(x, mask, gamask, output_attentions=output_attentions) # NOTE: only return the first output
        else:
            raise Exception("DA must be True! Please set `DA_M_mode=none` to disable DA!")
        o = x[0][:, -1, :]        # [B, H]
        output = (o,)
        if output_attentions:
            output += (x[1],)
            return output
        else:
            return output[0]

    @profile
    def cal_loss(self, batch_data, batch_idx=None):
        batch_user, batch_seqs, batch_last_items = batch_data
        if batch_idx==0:
            seq_output, seq_attention = self.forward(batch_seqs, output_attentions=True)
            wandb.log({"attention": wandb.Histogram(seq_attention[0].flatten().detach().cpu().numpy())})
        else:
            seq_output = self.forward(batch_seqs)

        test_item_emb = self.emb_layer.token_emb.weight[:self.item_num + 1]
        logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
        loss = self.loss_func(logits, batch_last_items)

        if self.KCL:
            aug_seq1, aug_seq2 = self._cl4srec_aug(batch_seqs)
            seq_output1 = self.forward(aug_seq1)
            seq_output2 = self.forward(aug_seq2)
            cl_loss = self.lmd * self.info_nce(
                seq_output1, seq_output2, temp=self.tau, batch_size=aug_seq1.shape[0])
            loss_dict = {
                'rec_loss': loss.item(),
                'cl_loss': cl_loss.item(),
            }
            loss += cl_loss
        else:
            loss_dict = {
                'rec_loss': loss.item(),
            }
        return loss, loss_dict

    def cal_kg_loss(self, batch_data):
        h, r, pos_t, neg_t = batch_data
        # (kg_batch_size, relation_dim)
        r_embed = self.relation_embed[r]
        h_embed = self.entity_embed[h]              # (kg_batch_size, entity_dim)
        pos_t_embed = self.entity_embed[pos_t]      # (kg_batch_size, entity_dim)
        neg_t_embed = self.entity_embed[neg_t]      # (kg_batch_size, entity_dim)
        h_item_idx = torch.where(h <= self.item_num)[0]
        h_embed[h_item_idx] = self.emb_layer.token_emb(h[h_item_idx])
        pos_t_item_idx = torch.where(pos_t <= self.item_num)[0]
        pos_t_embed[pos_t_item_idx] = self.emb_layer.token_emb(pos_t[pos_t_item_idx])
        neg_t_item_idx = torch.where(neg_t <= self.item_num)[0]
        neg_t_embed[neg_t_item_idx] = self.emb_layer.token_emb(neg_t[neg_t_item_idx])

        pos_score = torch.sum(
            torch.pow(h_embed + r_embed - pos_t_embed, 2), dim=1)     # (kg_batch_size)
        neg_score = torch.sum(
            torch.pow(h_embed + r_embed - neg_t_embed, 2), dim=1)     # (kg_batch_size)
        kg_loss = (-1.0) * F.logsigmoid(neg_score - pos_score)
        kg_loss = torch.mean(kg_loss)

        l2_loss = _L2_loss_mean(h_embed) + _L2_loss_mean(r_embed) + \
            _L2_loss_mean(pos_t_embed) + _L2_loss_mean(neg_t_embed)
        loss = kg_loss + 1e-3 * l2_loss
        return loss

    def full_predict(self, batch_data):
        with torch.no_grad():
            batch_user, batch_seqs, _ = batch_data
            logits = self.forward(batch_seqs)
            test_item_emb = self.emb_layer.token_emb.weight[:self.item_num + 1]
            scores = torch.matmul(logits, test_item_emb.transpose(0, 1))
        return scores
