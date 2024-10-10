import numpy as np
import pandas as pd
import torch
import os
import random
from torch.utils.data import Dataset, DataLoader
import networkx as nx

class NeighborFinder:
    def __init__(self, ratings):

        ratings = ratings.reset_index(drop=True)
        self.ratings = np.array(ratings)
        # print(self.ratings.shape)
        users = ratings['user_id'].unique()
        items = ratings['item_id'].unique()
        self.user_edgeidx = {cur_user: np.array(ratings[ratings.user_id == cur_user].index.tolist()) for cur_user in
                             users}  # 用户的边ID集合
        self.item_edgeidx = {cur_item: np.array(ratings[ratings.item_id == cur_item].index.tolist()) for cur_item in
                             items}  # item的边ID集合

    def get_user_neighbor(self, source_idx, timestamps, n_neighbors, device):

        assert (len(source_idx) == len(timestamps))

        adj_user = torch.zeros((len(source_idx), n_neighbors), dtype=torch.int32).to(device)  # 表示每一个节点的邻居向量
        user_mask = torch.ones((len(source_idx), n_neighbors), dtype=torch.bool).to(device)
        user_time = torch.zeros((len(source_idx), n_neighbors), dtype=torch.int32).to(
            device)  # time matirx，节点与其他max_nodes的时间差
        adj_user_edge = torch.zeros((len(source_idx), n_neighbors), dtype=torch.int32).to(device)

        edge_idxs = torch.searchsorted(self.ratings[:, 2], timestamps)

        for i in range(len(source_idx)):
            idx = torch.searchsorted(self.user_edgeidx[source_idx[i].item()], edge_idxs[i].item())  # 当前用户最近的边
            his_len = len(self.user_edgeidx[source_idx[i].item()][:idx])
            used_len = his_len if his_len <= n_neighbors else n_neighbors

            adj_user[i, n_neighbors - used_len:] = self.ratings[:, 1][
                self.user_edgeidx[source_idx[i].item()][idx - used_len:idx]]
            user_time[i, n_neighbors - used_len:] = self.ratings[:, 2][
                self.user_edgeidx[source_idx[i].item()][idx - used_len:idx]]
            user_mask[i, n_neighbors - used_len:] = 0
            adj_user_edge[i, n_neighbors - used_len:] = self.user_edgeidx[source_idx[i].item()][idx - used_len:idx]

        return adj_user, adj_user_edge, user_time, user_mask

    def get_item_neighbor(self, destination_idx, timestamps, n_neighbors, device):

        assert (len(destination_idx) == len(timestamps))

        adj_item = torch.zeros((len(destination_idx), n_neighbors), dtype=torch.int32).to(device)  # 表示每一个节点的邻居向量
        item_mask = torch.ones((len(destination_idx), n_neighbors), dtype=torch.bool).to(device)
        item_time = torch.zeros((len(destination_idx), n_neighbors), dtype=torch.int32).to(device)
        adj_item_edge = torch.zeros((len(destination_idx), n_neighbors), dtype=torch.int32).to(device)

        edge_idxs = torch.searchsorted(self.ratings[:, 2], timestamps)

        for i in range(len(destination_idx)):
            idx = torch.searchsorted(self.item_edgeidx[destination_idx[i].item()], edge_idxs[i].item())  # 当前用户最近的边
            his_len = len(self.item_edgeidx[destination_idx[i].item()][:idx])
            used_len = his_len if his_len <= n_neighbors else n_neighbors

            adj_item[i, n_neighbors - used_len:] = self.ratings[:, 0][
                self.item_edgeidx[destination_idx[i].item()][idx - used_len:idx]]
            item_time[i, n_neighbors - used_len:] = self.ratings[:, 2][
                self.item_edgeidx[destination_idx[i].item()][idx - used_len:idx]]
            item_mask[i, n_neighbors - used_len:] = 0
            adj_item_edge[i, n_neighbors - used_len:] = self.item_edgeidx[destination_idx[i].item()][idx - used_len:idx]

        return adj_item, adj_item_edge, item_time, item_mask

    def get_user_neighbor_ind(self, source_idx, edge_idx, n_neighbors, device):
        adj_user = np.zeros((len(edge_idx), n_neighbors), dtype=np.int32)  # 表示每一个节点的邻居向量
        user_mask = np.ones((len(edge_idx), n_neighbors), dtype=np.bool)
        user_time = np.zeros((len(edge_idx), n_neighbors), dtype=np.int32)  # time matirx，节点与其他max_nodes的时间差
        adj_user_edge = np.zeros((len(source_idx), n_neighbors), dtype=np.int32)

        for i in range(len(edge_idx)):
            idx = np.searchsorted(self.user_edgeidx[source_idx[i]], edge_idx[i]) + 1
            his_len = len(self.user_edgeidx[source_idx[i]][:idx])
            used_len = his_len if his_len <= n_neighbors else n_neighbors

            try:
                adj_user[i, n_neighbors - used_len:] = self.ratings[:, 1][
                    self.user_edgeidx[source_idx[i].item()][idx - used_len:idx]]
            except:
                print(i)
                print(idx, used_len, n_neighbors, self.user_edgeidx[source_idx[i]].shape, self.ratings[:, 1].shape)
                print(self.user_edgeidx[source_idx[i]])
                print(self.ratings[:, 1])
            user_time[i, n_neighbors - used_len:] = self.ratings[:, 2][
                self.user_edgeidx[source_idx[i].item()][idx - used_len:idx]]
            user_mask[i, n_neighbors - used_len:] = 0
            adj_user_edge[i, n_neighbors - used_len:] = self.user_edgeidx[source_idx[i].item()][idx - used_len:idx]
            
        return torch.from_numpy(adj_user).to(device), torch.from_numpy(adj_user_edge).to(device), torch.from_numpy(
            user_time).to(device), torch.from_numpy(user_mask).to(device)

    def get_item_neighbor_ind(self, destination_idx, edge_idx, n_neighbors, device):

        adj_item = np.zeros((len(destination_idx), n_neighbors), dtype=np.int32)  # 表示每一个节点的邻居向量
        item_mask = np.ones((len(destination_idx), n_neighbors), dtype=np.bool)
        item_time = np.zeros((len(destination_idx), n_neighbors), dtype=np.int32)
        adj_item_edge = np.zeros((len(destination_idx), n_neighbors), dtype=np.int32)

        for i in range(len(destination_idx)):
            idx = np.searchsorted(self.item_edgeidx[destination_idx[i]], edge_idx[i]) + 1
            his_len = len(self.item_edgeidx[destination_idx[i]][:idx])
            used_len = his_len if his_len <= n_neighbors else n_neighbors

            adj_item[i, n_neighbors - used_len:] = self.ratings[:,0][self.item_edgeidx[destination_idx[i]][idx - used_len:idx]]
            item_time[i, n_neighbors - used_len:] = self.ratings[:,2][self.item_edgeidx[destination_idx[i]][idx - used_len:idx]]
            item_mask[i, n_neighbors - used_len:] = 0
            adj_item_edge[i, n_neighbors - used_len:] = self.item_edgeidx[destination_idx[i]][idx - used_len:idx]

        return torch.from_numpy(adj_item).to(device), torch.from_numpy(adj_item_edge).to(device), torch.from_numpy(
            item_time).to(device), torch.from_numpy(item_mask).to(device)
        
    def get_user_neighbor_ind_slice(self, source_idx, edge_idx, n_neighbors, device):
        adj_user = np.zeros((len(edge_idx), n_neighbors), dtype=np.int32)  # 表示每一个节点的邻居向量
        user_mask = np.ones((len(edge_idx), n_neighbors), dtype=np.bool)
        user_time = np.zeros((len(edge_idx), n_neighbors), dtype=np.int32)  # time matirx，节点与其他max_nodes的时间差
        adj_user_edge = np.zeros((len(source_idx), n_neighbors), dtype=np.int32)

        for i in range(len(edge_idx)):
            idx = np.searchsorted(self.user_edgeidx[source_idx[i]], edge_idx[i]) + 1
            his_len = len(self.user_edgeidx[source_idx[i]][:idx])
            used_len = min(his_len, n_neighbors)  # 保证used_len不超过n_neighbors

            # 确保数组长度匹配
            adj_user_slice_length = len(adj_user[i, n_neighbors - used_len:])
            ratings_slice = self.ratings[:, 1][self.user_edgeidx[source_idx[i].item()][idx - used_len:idx]]
            adj_user[i, n_neighbors - used_len:] = ratings_slice[:adj_user_slice_length]

            user_time_slice = self.ratings[:, 2][self.user_edgeidx[source_idx[i].item()][idx - used_len:idx]]
            user_time[i, n_neighbors - used_len:] = user_time_slice[:adj_user_slice_length]

            adj_user_edge_slice = self.user_edgeidx[source_idx[i].item()][idx - used_len:idx]
            adj_user_edge[i, n_neighbors - used_len:] = adj_user_edge_slice[:adj_user_slice_length]

            user_mask[i, n_neighbors - used_len:] = 0
            
        return torch.from_numpy(adj_user).to(device), torch.from_numpy(adj_user_edge).to(device), torch.from_numpy(
            user_time).to(device), torch.from_numpy(user_mask).to(device)
        
        
    def get_item_neighbor_ind_slice(self, destination_idx, edge_idx, n_neighbors, device):
        adj_item = np.zeros((len(destination_idx), n_neighbors), dtype=np.int32)  # Neighbor vector for each node
        item_mask = np.ones((len(destination_idx), n_neighbors), dtype=np.bool)
        item_time = np.zeros((len(destination_idx), n_neighbors), dtype=np.int32)  # Time difference matrix
        adj_item_edge = np.zeros((len(destination_idx), n_neighbors), dtype=np.int32)

        for i in range(len(destination_idx)):
            idx = np.searchsorted(self.item_edgeidx[destination_idx[i]], edge_idx[i]) + 1
            his_len = len(self.item_edgeidx[destination_idx[i]][:idx])
            used_len = min(his_len, n_neighbors)  # Ensure used_len does not exceed n_neighbors

            # Ensure array lengths match before assignment
            adj_item_slice_length = len(adj_item[i, n_neighbors - used_len:])
            ratings_slice = self.ratings[:, 0][self.item_edgeidx[destination_idx[i]][idx - used_len:idx]]
            adj_item[i, n_neighbors - used_len:] = ratings_slice[:adj_item_slice_length]

            item_time_slice = self.ratings[:, 2][self.item_edgeidx[destination_idx[i]][idx - used_len:idx]]
            item_time[i, n_neighbors - used_len:] = item_time_slice[:adj_item_slice_length]

            adj_item_edge_slice = self.item_edgeidx[destination_idx[i]][idx - used_len:idx]
            adj_item_edge[i, n_neighbors - used_len:] = adj_item_edge_slice[:adj_item_slice_length]

            item_mask[i, n_neighbors - used_len:] = 0  # Adjust mask as required

        return (
            torch.from_numpy(adj_item).to(device),
            torch.from_numpy(adj_item_edge).to(device),
            torch.from_numpy(item_time).to(device),
            torch.from_numpy(item_mask).to(device)
        )


def data_partition(fname):
    if "Beauty" in fname:
        ratings = pd.read_csv(fname,sep='\t')
        ratings = ratings.iloc[:,:4]
        # print(ratings.keys())
        ratings.rename(columns={
            'user_id:token': 'user_id',
            'item_id:token': 'item_id',
            'rating:float': 'rating',
            'timestamp:float': 'timestamp'
        }, inplace=True)
        # print(ratings.keys())
    else:
        ratings = []
        with open(os.path.join(fname, 'ratings.dat')) as f:
            for l in f:
                user_id, item_id, rating, timestamp = [_ for _ in l.split('::')]
                rating = float(rating)
                timestamp = int(timestamp)
                ratings.append({
                        'user_id': user_id,
                        'item_id': item_id,
                        'rating': rating,
                        'timestamp': timestamp,
                        })
        ratings = pd.DataFrame(ratings)
    users = ratings['user_id'].unique() # 所有用户的id
    items = ratings['item_id'].unique() # 所有item的id 
     
    ratings['timestamp'] = ratings['timestamp'] - min(ratings['timestamp']) # 每条边的时间戳相当于初始时间戳过了多少时间
    
    # 下面这个循环用于去掉出现次数小于5次的用户和项目
    for i in range(1000):
        item_count = ratings['item_id'].value_counts()
        item_count.name = 'item_count'
        ratings = ratings.join(item_count, on='item_id')

        user_count = ratings['user_id'].value_counts()
        user_count.name = 'user_count'
        ratings = ratings.join(user_count, on='user_id')
        ratings = ratings[(ratings['user_count'] >= 10) & (ratings['item_count'] >= 10)]

        if len(ratings['user_id'].unique()) == len(users) and len(ratings['item_id'].unique()) == len(items):
            break
        users = ratings['user_id'].unique()
        items = ratings['item_id'].unique()
        del ratings['user_count']
        del ratings['item_count']
    
    del ratings['user_count']
    del ratings['item_count']
    
    users = ratings['user_id'].unique()
    items = ratings['item_id'].unique()
    
    item_count = ratings['item_id'].value_counts()
    item_count.name = 'item_count'
    ratings = ratings.join(item_count, on='item_id')

    user_count = ratings['user_id'].value_counts()
    user_count.name = 'user_count'
    ratings = ratings.join(user_count, on='user_id')
    ratings = ratings[(ratings['user_count'] >= 10) & (ratings['item_count'] >= 10)]

    users = ratings['user_id'].unique()
    items = ratings['item_id'].unique()

    # 下面这几行代码相当于对用户和项目重新分配了id
    # 为了方便理解，可以理解成tx_id2node_id
    user_ids_invmap = {id_: i for i, id_ in enumerate(users)}
    item_ids_invmap = {id_: i for i, id_ in enumerate(items)}
 
    # ratings['user_id'].replace(user_ids_invmap, inplace=True)
    # ratings['item_id'].replace(item_ids_invmap, inplace=True)
    
    ratings['user_id'] = ratings['user_id'].map(user_ids_invmap)
    ratings['item_id'] = ratings['item_id'].map(item_ids_invmap)


    print('user_count:'+str(len(users))+','+'item_count:'+str(len(items)))
    print('avr of user:'+str(ratings['user_id'].value_counts().mean())+'avr of item:'+str(ratings['item_id'].value_counts().mean()))
    print(len(ratings))

    users = ratings['user_id'].unique()
    items = ratings['item_id'].unique()
    
    # 这一步主要是根据时间戳进行交互的排序
    ratings = ratings.sort_values(by='timestamp',ascending=True)  
    
    ratings = ratings.reset_index(drop=True)
    
    
    full_data = []
    
    # 这里是记录了每个用户和每个项目各自参与了哪些交互，注意，list中保存的是交互的索引，而不是用户或者项目的id
    adj_user = {cur_user:ratings[ratings.user_id == cur_user].index.tolist() for cur_user in users} 
    adj_item = {cur_item:ratings[ratings.item_id == cur_item].index.tolist() for cur_item in items}
    
    for i in range(ratings.shape[0]):  #edge ID
        
        cur_user = ratings['user_id'].iloc[i]
        cur_item = ratings['item_id'].iloc[i]
        #确保训练集和测试集中的序列至少含有3个邻居
        #这里没看懂，总之就是每个用户或者项目的前三个交互被省略了
        if adj_user[cur_user].index(i)>=3 and adj_item[cur_item].index(i)>=3:
            full_data.append(i)
          
    offset1 = int(len(full_data) * 0.8)
    offset2 = int(len(full_data) * 0.9)
    random.shuffle(full_data)
    train_data, valid_data, test_data = full_data[0:offset1], full_data[offset1:offset2], full_data[offset2:len(full_data)]
   
    del ratings['rating']
    del ratings['user_count']
    del ratings['item_count']
    print(ratings.columns)
    
    return ratings, train_data, valid_data, test_data, user_ids_invmap, item_ids_invmap


class pre_train_dataset(Dataset):
    def __init__(self, train_data, num_users, num_items, positive_num, negative_num, weak_positive_num):
        self.train_data = train_data
        self.num_users = num_users
        self.num_items = num_items
        self.positive_num = positive_num
        self.negative_num = negative_num
        self.weak_positive_num = weak_positive_num
        self.make_graph()

    def make_graph(self):
        # print(self.train_data.shape)
        # print(self.train_data.columns)
        self.train_data.loc[:, 'item_id'] = self.train_data['item_id'] + self.num_users
        # self.pre_train_graph = np.array(self.train_data[['user_id', 'item_id']].values.tolist(), dtype=np.int32)
        # print(self.pre_train_graph.shape)
        
        self.users = self.train_data['user_id'].unique()
        self.items = self.train_data['item_id'].unique() 
        # print(min(self.users), max(self.users))
        # print(min(self.items), max(self.items))
        # print(self.num_items, self.num_users)
        # print(type(self.items))
        
        pre_train_graph = nx.Graph()
        pre_train_graph.add_nodes_from(self.users)
        pre_train_graph.add_nodes_from(self.items)
        self.pre_train_edges = np.array(self.train_data[['user_id', 'item_id']].values.tolist())
        pre_train_graph.add_edges_from(np.array(self.train_data[['user_id', 'item_id']].values.tolist()))
        self.pre_train_graph = pre_train_graph
        
        degree_dict = dict(self.pre_train_graph.degree)
        item_degrees = {node: degree for node, degree in degree_dict.items() if node in self.items}
        sorted_items = sorted(item_degrees.items(), key=lambda x: x[1], reverse=True)
        self.popular_items = [item[0] for item in sorted_items]

        

    def find_second_order_neighbors(self, node):
        # 获取 node 的一阶邻居
        first_order_neighbors = set(nx.neighbors(self.pre_train_graph, node))
        # 获取一阶邻居的邻居（即二阶邻居）
        second_order_neighbors = set()
        for neighbor in first_order_neighbors:
            second_order_neighbors.update(set(nx.neighbors(self.pre_train_graph, neighbor)))
        # 移除原节点和其一阶邻居，只保留真正的二阶邻居
        second_order_neighbors.discard(node)
        second_order_neighbors.difference_update(first_order_neighbors)
        return list(second_order_neighbors)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        anchor_item = self.items[idx]
        poitive_item_neighbors = self.find_second_order_neighbors(anchor_item)
        random.shuffle(poitive_item_neighbors)
        if len(poitive_item_neighbors) >= self.positive_num:
            poitive_item_neighbors = poitive_item_neighbors[:self.positive_num]
        else:
            poitive_item_neighbors = poitive_item_neighbors + [anchor_item] * (self.positive_num - len(poitive_item_neighbors))
        
        weak_popular_items = []
        for item in self.popular_items:
            if item != anchor_item and item not in poitive_item_neighbors:
                weak_popular_items.append(item)
                if len(weak_popular_items) >= self.weak_positive_num:
                    break
        # if len(weak_popular_items) < self.weak_popular_num:
        
        negative_item_neighbors = set(self.items) - set(poitive_item_neighbors) - set(weak_popular_items) - set([anchor_item])
        negative_item_neighbors = list(negative_item_neighbors)
        random.shuffle(negative_item_neighbors)
        if len(negative_item_neighbors) >= self.negative_num:
            negative_item_neighbors = negative_item_neighbors[:self.negative_num]
        
        
        # print(anchor_item.shape, poitive_item_neighbors.shape, weak_popular_items.shape, negative_item_neighbors.shape)
        poitive_item_neighbors = torch.tensor(poitive_item_neighbors, dtype=torch.long)
        weak_popular_items = torch.tensor(weak_popular_items, dtype=torch.long)
        negative_item_neighbors = torch.tensor(negative_item_neighbors, dtype=torch.long)
        anchor_item = torch.tensor(anchor_item, dtype=torch.long)
        return anchor_item, poitive_item_neighbors, weak_popular_items, negative_item_neighbors

    
    
if __name__ == '__main__':
    ratings, train_data, valid_data, test_data = data_partition('data/movielens/ml-1m')