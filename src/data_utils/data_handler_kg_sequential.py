import torch
import torch.utils.data as data
import numpy as np
import pandas as pd
import scipy.sparse as sp
from config.configurator import configs
from os import path
from collections import defaultdict
from tqdm import tqdm
from .datasets_kg import KGTrainDataset, KGTestDataset, KGTripletDataset
from .datasets_sequential import SequentialDataset


class DataHandlerKGSequential:
    def __init__(self) -> None:
        if configs['data']['name'] == 'mind':
            predir = './datasets/kg/mind_kg/'
        elif configs['data']['name'] in [
            'tiny',
            'ml-100k', 'ml-1m', 'ml-10m', 'ml-20m', 'Amazon_Books', 'amazon-book', 'last-fm', 
            'ml-100k_v2', 'amazon-book_v2', 'last-fm_v2',
            'lfm1b-albums', 'lfm1b-artists', 'lfm1b-tracks'
        ]:
            predir = f'./datasets/kg_sequential/{configs["data"]["name"]}/'
        else: raise Exception("Please provide the correct dataset name")

        configs['data']['dir'] = predir
        self.trn_file = path.join(predir, 'train.txt')
        self.val_file = path.join(predir, 'test.txt')
        self.tst_file = path.join(predir, 'test.txt') 
        self.kg_file = path.join(predir, 'kg_final.csv')
        self.max_item_id = 0

    def _read_triplets(self, file_name):
        """ read the 'kg_final.csv' file 
        format: h_id, r_id, t_id -> index start from 1
        """
        can_triplets_np = np.loadtxt(file_name, dtype=np.int32)
        can_triplets_np = np.unique(can_triplets_np, axis=0)

        is_zero_start = min(min(can_triplets_np[:, 0]), min(can_triplets_np[:, 2])) == 0
        # 
        if is_zero_start:
            print("[NOTE] entity id in KG start from 0, add 1")
            can_triplets_np[:, 0] += 1
            can_triplets_np[:, 2] += 1
        is_zero_relation = min(can_triplets_np[:, 1]) == 0
        if is_zero_relation:
            print("[NOTE] relation id in KG start from 0, add 1")
            can_triplets_np[:, 1] += 1
        # get triplets with inverse direction like <entity, is-aspect-of, item>
        inv_triplets_np = can_triplets_np.copy()
        inv_triplets_np[:, 0] = can_triplets_np[:, 2]
        inv_triplets_np[:, 2] = can_triplets_np[:, 0]
        inv_triplets_np[:, 1] = can_triplets_np[:, 1] + max(can_triplets_np[:, 1]) # 原本从1开始这里就不用 + 1
        # consider two additional relations --- 'interact' and 'be interacted'
        can_triplets_np[:, 1] = can_triplets_np[:, 1] + 1
        inv_triplets_np[:, 1] = inv_triplets_np[:, 1] + 1
        # get full version of knowledge graph
        triplets = np.concatenate((can_triplets_np, inv_triplets_np), axis=0)

        n_entities = max(max(triplets[:, 0]), max(triplets[:, 2]))  # including items + users
        n_nodes = n_entities + configs['data']['user_num']
        n_relations = max(triplets[:, 1])

        configs['data']['entity_num'] = n_entities
        configs['data']['node_num'] = n_nodes
        configs['data']['relation_num'] = n_relations
        configs['data']['triplet_num'] = len(triplets)
        configs['data']['relation_num_raw'] = n_relations // 2
        configs['data']['triplet_num_raw'] = len(triplets) // 2

        return can_triplets_np, triplets

    def _build_kg_graph(self, triplets):
        """ from data_handler_kg.py """
        kg_dict = defaultdict(list)
        # h, t, r
        kg_edges = list()
        print("Begin to load knowledge graph triples ...")
        for h_id, r_id, t_id in tqdm(triplets, ascii=True):
            # h,t,r
            kg_edges.append([h_id, t_id, r_id])
            kg_dict[h_id].append((r_id, t_id))
        return kg_edges, kg_dict

    def _read_tsv_to_user_seqs(self, tsv_file):
        user_seqs = {"uid": [], "item_seq": [], "item_id": []}
        with open(tsv_file, 'r') as f:
            line = f.readline()
            line = f.readline()
            while line:
                uid, seq, last_item = line.strip().split('\t')
                seq = seq.split(' ')
                seq = [int(item) for item in seq]
                user_seqs["uid"].append(int(uid))
                user_seqs["item_seq"].append(seq)
                user_seqs["item_id"].append(int(last_item))

                self.max_item_id = max(
                    self.max_item_id, max(max(seq), int(last_item)))
                line = f.readline()
        return user_seqs

    def _read_seq_train_test(self, trn_file, tst_file):
        """ format:
        train: [uid, iids]
        test: [uid, iid]
        """
        uid2seq = {}
        mn_iid = 1e9
        mx_iid = 0
        with open(trn_file, 'r') as f:
            for line in f:
                ids = line.strip().split(' ')
                ids = list(map(int, ids))
                uid2seq[ids[0]] = ids[1:]
                mn_iid = min(mn_iid, min(ids[1:]))
                mx_iid = max(mx_iid, max(ids[1:]))
        uid2label = {}
        with open(tst_file, 'r') as f:
            for line in f:
                ids = line.strip().split(' ')
                ids = list(map(int, ids))
                assert len(ids) == 2
                uid2label[ids[0]] = ids[1]
                mn_iid = min(mn_iid, ids[1])
                mx_iid = max(mx_iid, ids[1])
        if mn_iid == 0:
            print("[NOTE] item ids start from 0, add 1")
            for uid in uid2seq:
                uid2seq[uid] = [i+1 for i in uid2seq[uid]]
            for uid in uid2label:
                uid2label[uid] += 1
            mx_iid += 1
        user_seqs_train = {"uid": [], "item_seq": [], "item_id": []}
        user_seqs_test = {"uid": [], "item_seq": [], "item_id": []}
        for uid, seq in uid2seq.items():
            user_seqs_train["uid"].append(uid)
            user_seqs_train["item_seq"].append(seq[:-1])
            user_seqs_train["item_id"].append(seq[-1])
        for uid, seq in uid2seq.items():
            user_seqs_test["uid"].append(uid)
            user_seqs_test["item_seq"].append(seq)
            user_seqs_test["item_id"].append(uid2label[uid])
        self.max_item_id = mx_iid
        return user_seqs_train, user_seqs_test

    def _set_statistics(self, user_seqs_train, user_seqs_test):
        user_num = max(max(user_seqs_train["uid"]), max(user_seqs_test["uid"])) + 1
        interaction_num = 0
        for seq in user_seqs_test["item_seq"]:
            interaction_num += len(seq) + 1
        configs['data']['user_num'] = user_num
        configs['data']['item_num'] = self.max_item_id + 1
        configs['data']['interaction_num'] = interaction_num

    def _seq_aug(self, user_seqs):
        user_seqs_aug = {"uid": [], "item_seq": [], "item_id": []}
        for uid, seq, last_item in zip(user_seqs["uid"], user_seqs["item_seq"], user_seqs["item_id"]):
            user_seqs_aug["uid"].append(uid)
            user_seqs_aug["item_seq"].append(seq)
            user_seqs_aug["item_id"].append(last_item)
            for i in range(1, len(seq)-1):
                user_seqs_aug["uid"].append(uid)
                user_seqs_aug["item_seq"].append(seq[:i])
                user_seqs_aug["item_id"].append(seq[i])
        return user_seqs_aug

    def _build_edges(self, kg_triplets):
        arr = np.array(kg_triplets)
        head, tail, rel = arr[:, 0], arr[:, 2], arr[:, 1]
        head, tail, rel = torch.LongTensor(head), torch.LongTensor(tail), torch.LongTensor(rel)
        return (head, tail), rel

    def filter_high_degree_nodes(self, kg_origin, threshold=300):
        def stat_node_degree(kg):
            # stat node degree
            node_degree = kg.groupby('h').size().reset_index(name='degree')
            node_degree = node_degree.sort_values(by=['degree'], ascending=False)
            return node_degree
        def filter_kg(kg, node_set):
            # filter kg
            kg = kg[~(kg['h'].isin(node_set) | kg['t'].isin(node_set))]
            return kg
        kg_df = pd.DataFrame(kg_origin, columns=['h', 'r', 't'])
        node_degree = stat_node_degree(kg_df)
        filter_node_set = node_degree[node_degree['degree'] > threshold]['h'].values.tolist()
        kg_filtered = filter_kg(kg_df, filter_node_set)
        return kg_filtered.values

    def load_data(self):
        user_seqs_train, user_seqs_test = self._read_seq_train_test(self.trn_file, self.tst_file)
        self._set_statistics(user_seqs_train, user_seqs_test)
         
        # 1] KG
        kg_triplets_raw, kg_triplets = self._read_triplets(self.kg_file)
        if 'filter' in configs['data'] and configs['data']['filter']:
            kg_triplets = self.filter_high_degree_nodes(kg_triplets, configs['data']['filter_threshold'])
        self.kg_edges, self.kg_dict = self._build_kg_graph(kg_triplets)
        self.kg_edges_raw, self.kg_dict_raw = self._build_kg_graph(kg_triplets_raw)
        self.edge_index, self.edge_type = self._build_edges(kg_triplets)

        # seqeuntial augmentation: [1, 2, 3,] -> [1,2], [3]
        if 'seq_aug' in configs['data'] and configs['data']['seq_aug']:
            print(f"augmenting user sequences ...")
            user_seqs_aug = self._seq_aug(user_seqs_train)
            print(f"augment done!")
            trn_data = SequentialDataset(user_seqs_train, user_seqs_aug=user_seqs_aug)
        else:
            trn_data = SequentialDataset(user_seqs_train)
        tst_data = SequentialDataset(user_seqs_test, mode='test')
        self.test_dataloader = data.DataLoader(
            tst_data, batch_size=configs['test']['batch_size'], shuffle=False, num_workers=0)
        self.train_dataloader = data.DataLoader(
            trn_data, batch_size=configs['train']['batch_size'], shuffle=True, num_workers=0)
        trn_data_raw = SequentialDataset(user_seqs_train, mode='test')
        self.train_dataloader_raw = data.DataLoader(
            trn_data_raw, batch_size=configs['train']['batch_size'], shuffle=False, num_workers=0)

        if configs['model']['name'] in ['kgat', 'ourkatcl', 'katrec', 'ourshi']:
            triplet_data = KGTripletDataset(kg_triplets, self.kg_dict)
            # no shuffle because of randomness
            self.triplet_dataloader = data.DataLoader(
                triplet_data, batch_size=configs['train']['kg_batch_size'], shuffle=False, 
            )
        # for GES
        if configs['model']['name']=='ges':
            self.adj_matrix = self.get_GES_adj_matrix(item_seqs=user_seqs_train['item_seq'], num_item=configs['data']['item_num']+2, alpha=1.0)
    

    def get_GES_adj_matrix(self, item_seqs, num_item, alpha, max_len=int(1e9)):
        row_seq = [seq[-max_len:][n] for seq in item_seqs for n in range(len(seq[-max_len:])-1)] + [seq[-max_len:][n+1] for seq in item_seqs for n in range(len(seq[-max_len:])-1)]
        col_seq = [seq[-max_len:][n+1] for seq in item_seqs for n in range(len(seq[-max_len:])-1)] + [seq[-max_len:][n] for seq in item_seqs for n in range(len(seq[-max_len:])-1)]
        rel_matrix = sp.coo_matrix(([alpha]*len(row_seq), (row_seq, col_seq)), (num_item, num_item)).astype(np.float32) + sp.eye(num_item)
        row_sum = np.array(rel_matrix.sum(1)) + 1e-24
        degree_mat_inv_sqrt = sp.diags(np.power(row_sum, -0.5).flatten())
        rel_matrix_normalized = degree_mat_inv_sqrt.dot(rel_matrix.dot(degree_mat_inv_sqrt)).tocoo()
        indices = np.vstack((rel_matrix_normalized.row, rel_matrix_normalized.col)).transpose()
        values = rel_matrix_normalized.data.astype(np.float32)
        shape = rel_matrix_normalized.shape

        adj_matrix = torch.sparse_coo_tensor(torch.tensor(indices, dtype=torch.int64).T, torch.tensor(values), torch.Size(shape))
        return adj_matrix