from VQ_DR.deezer.utils import *
import pickle
import torch
import numpy as np
from copy import deepcopy
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

class TDMRecDataset(RecDataset):
    def __init__(self, n_user, n_item, user_features, target_items, mode, depth, neg_num=1):
        self.n_user = n_user
        self.n_item = n_item
        self.user_features = user_features
        self.target_items_per_layer = target_items
        self.mode = mode
        self.neg_num = neg_num
        self.depth = depth

    def __getitem__(self, index):
        depth = np.random.randint(0, self.depth)
            
        if self.mode == 'train':
            neg_item = torch.randint((1 << (depth+1)) - 1, min((1 << (depth+2)) - 1, self.n_item), (self.neg_num,))
            pos_item = np.random.choice(self.target_items_per_layer[index][depth], 1)
            return self.user_features[index], torch.IntTensor(pos_item), neg_item
        else:
            raise ValueError('TDMRecDataset can only used in train mode')

    def __len__(self):
        return self.user_features.shape[0]
    
def items_2_id(songs_list, tree_data):
    nodes_list = []
    for songs in songs_list:
        nodes_list.append([tree_data['item_2_id'][song] for song in songs])
    return nodes_list

def extend_layers(leafs_list, tree_data):
    res = []
    for leaf in leafs_list:
        leaf = np.array(leaf)
        nodes_per_layer = []
        for i in range(tree_data['depth']):
            #leaf = np.unique(leaf)
            nodes_per_layer.append(deepcopy(leaf))
            leaf = (leaf-1)//2
        nodes_per_layer.reverse()
        res.append(nodes_per_layer)
    return res
    

def generate_data_by_user(dataset, data_dir, batch_size, neg_num, inference_batch_size, tree_data):
    if dataset == 'deezer':
        node_embedding_pth, train_songs_pth, train_features_pth, valid_songs_pth, valid_features_pth, \
        test_songs_pth, test_features_pth = data_dir
    node_embedding = torch.FloatTensor(np.load(node_embedding_pth))
    train_songs = pickle.load(open(train_songs_pth, 'rb'))
    train_features = pickle.load(open(train_features_pth, 'rb'))
    valid_songs = pickle.load(open(valid_songs_pth, 'rb'))
    valid_features = pickle.load(open(valid_features_pth, 'rb'))
    test_songs = pickle.load(open(test_songs_pth, 'rb'))
    test_features = pickle.load(open(test_features_pth, 'rb'))
    #print('extend begin')
    #t1 = time.time()
    #print(node_embedding[tree_data['id_2_index'][tree_data['item_2_id'][0]]])
    train_nodes = items_2_id(train_songs, tree_data)
    train_nodes = extend_layers(train_nodes, tree_data=tree_data)
    valid_nodes = items_2_id(valid_songs, tree_data)
    test_nodes = items_2_id(test_songs, tree_data)
    with open('train_nodes.pkl', 'wb') as f:
        pickle.dump(train_nodes, f)
    with open('valid_nodes.pkl', 'wb') as f:
        pickle.dump(valid_nodes, f)
    with open('test_nodes.pkl', 'wb') as f:
        pickle.dump(test_nodes, f)
        
    with open('train_nodes.pkl', 'rb') as f:
        train_nodes = pickle.load(f)
    with open('valid_nodes.pkl', 'rb') as f:
        valid_nodes = pickle.load(f)
    with open('test_nodes.pkl', 'rb') as f:
        test_nodes = pickle.load(f)
    #print('extend end', time.time() - t1)
    
    n_user = train_features.shape[0] + valid_features.shape[0] + test_features.shape[0]
    n_item = node_embedding.shape[0]
    user_input_dim = train_features.shape[1]
    print("Dataset: {} n_user:{} n_item:{}".format(dataset, n_user, n_item))
    # create dataset
    train_dataset = TDMRecDataset(n_user=n_user, n_item=n_item, user_features=train_features,
                               target_items=train_nodes, neg_num=neg_num, depth=tree_data['depth'], mode='train')
    valid_dataset = RecDataset(n_user=n_user, n_item=n_item, user_features=valid_features,
                               target_items=valid_nodes, mode='valid')
    test_dataset = RecDataset(n_user=n_user, n_item=n_item, user_features=test_features,
                              target_items=test_nodes, mode='test')
    # create data loader
    train_songs_len = np.asarray([len(i) for i in train_songs])
    train_songs_dist = torch.from_numpy(train_songs_len / sum(train_songs_len))
    weight_sampler = WeightedRandomSampler(train_songs_dist, len(train_songs_dist))
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=18,
                              sampler=weight_sampler)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=inference_batch_size, shuffle=False,
                              collate_fn=collate_fn)
    test_loader = DataLoader(dataset=test_dataset, batch_size=inference_batch_size, shuffle=False,
                             collate_fn=collate_fn)
    return train_loader, valid_loader, test_loader, node_embedding, n_user, n_item, user_input_dim