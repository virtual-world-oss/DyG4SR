import time
import torch
import numpy as np
import random
from torch.utils.data import DataLoader

from data_prepare import data_partition,NeighborFinder
from model import PTGCN
from modules import TimeEncode,MergeLayer,time_encoding
import pickle as pkl
import os
from tqdm import tqdm

# from utils import get_ddygs

class Config(object):
    """config."""
    # data = 'Moivelens'
    data = 'beauty'
    data_path = './data/beauty'
    data_raw = False
    batch_size = 64 # batch_size = 256, origin=64, [64,128,256,1024]
    n_degree = [20,50]  #'Number of neighbors to sample'
    num_shots = 30 # num_shots = 2, origin=2, [2,5,10]
    n_head = 4  #'Number of heads used in attention layer'
    n_epoch = 50 # n_epoch = 100 #'Number of epochs'
    n_layer = 2 #'Number of network layers'
    lr = 0.0001 # lr = 0.0005  #'Learning rate' origin=0.0001,[0.00003,0.0001,0.0003,0.0005]
    patience = 25  #'Patience for early stopping' origin=25,[5,25]
    drop_out = 0.1  #'Dropout probability' origin=0.1, [0.1,0.3]
    gpu = 0,  #'Idx for the gpu to use'
    node_dim = 160  #'Dimensions of the node embedding' origin=160, [160,64]
    time_dim = 160  #'Dimensions of the time embedding' origin=160, [160,64]
    embed_dim = 160 #'Dimensions of the hidden embedding' origin=160, [160,64]
    is_GPU = True
    temperature = 0.07
    valid_batch_size = 64
    test_batch_size = 64
    
def evaluate_val(model, ratings, items, dl, adj_user_edge, adj_item_edge, adj_user_time, adj_item_time, device):
     # 准备工作
    torch.cuda.empty_cache()
    NDCG5 = 0.0
    NDCG10 = 0.0
    recall5 = 0.0
    recall10 =0.0
    num_sample = 0
    
    with torch.no_grad():
        model = model.eval()
        
        # for ix,batch in tqdm(enumerate(dl), total=len(dl), desc="validation"):
        for ix,batch in enumerate(dl):
            #if ix%100==0:
               # print('batch:',ix)
            count = len(batch)
            num_sample = num_sample + count
            b_user_edge = find_latest_1D(np.array(ratings.iloc[batch]['user_id']), adj_user_edge, adj_user_time, ratings.iloc[batch]['timestamp'].tolist())
            b_user_edge = torch.from_numpy(b_user_edge).to(device)
            b_users = torch.from_numpy(np.array(ratings.iloc[batch]['user_id'])).to(device) 
            
            b_item_edge = find_latest_1D(np.array(ratings.iloc[batch]['item_id']), adj_item_edge, adj_item_time, ratings.iloc[batch]['timestamp'].tolist())
            b_item_edge = torch.from_numpy(b_item_edge).to(device)
            b_items = torch.from_numpy(np.array(ratings.iloc[batch]['item_id'])).to(device)
            timestamps = torch.from_numpy(np.array(ratings.iloc[batch]['timestamp'])).to(device)
            
            negative_samples = sampler(items, adj_user, ratings.iloc[batch]['user_id'].tolist() ,100)  
            neg_edge = find_latest(negative_samples, adj_item_edge, adj_item_time, ratings.iloc[batch]['timestamp'].tolist())
            negative_samples = torch.from_numpy(np.array(negative_samples)).to(device)
            item_set = torch.cat([b_items.view(-1,1),negative_samples], dim=1) #batch, 101
            timestamps_set = timestamps.unsqueeze(1).repeat(1,101)
            neg_edge = torch.from_numpy(neg_edge).to(device)
            edge_set = torch.cat([b_item_edge.view(-1,1),neg_edge], dim=1) #batch, 101
            
            user_embeddings = model(b_users, b_user_edge,timestamps, config.n_layer, nodetype='user')
            itemset_embeddings = model(item_set.flatten(), edge_set.flatten(), timestamps_set.flatten(), config.n_layer, nodetype='item')
            itemset_embeddings = itemset_embeddings.view(count, 101, -1)
            
            logits = torch.bmm(user_embeddings.unsqueeze(1), itemset_embeddings.permute(0,2,1)).squeeze(1) # [count,101]
            logits = -logits.cpu().numpy()
            rank = logits.argsort().argsort()[:,0]
            
            recall5 += np.array(rank<5).astype(float).sum()
            recall10 += np.array(rank<10).astype(float).sum()
            NDCG5 += (1 / np.log2(rank + 2))[rank<5].sum()
            NDCG10 += (1 / np.log2(rank + 2))[rank<10].sum()
            
        recall5 = recall5/num_sample
        recall10 = recall10/num_sample
        NDCG5 = NDCG5/num_sample
        NDCG10 = NDCG10/num_sample
            
        print("===> recall_5: {:.10f}, recall_10: {:.10f}, NDCG_5: {:.10f}, NDCG_10: {:.10f}, time:{}".format(recall5, recall10, NDCG5, NDCG10, time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))))

    return recall5, recall10, NDCG5, NDCG10

def evaluate(model, ratings, items, dl, adj_user_edge, adj_item_edge, adj_user_time, adj_item_time, device):
    # 准备工作
    torch.cuda.empty_cache()
    NDCG5 = 0.0
    NDCG10 = 0.0
    recall5 = 0.0
    recall10 =0.0
    num_sample = 0
    
    test_batch_size = 150
    
    with torch.no_grad():
        model = model.eval()
        
        for ix,batch in tqdm(enumerate(dl), total=len(dl), desc="test"):
            #if ix%100==0:
               # print('batch:',ix)
            count = len(batch)
            num_sample = num_sample + count
            b_user_edge = find_latest_1D(np.array(ratings.iloc[batch]['user_id']), adj_user_edge, adj_user_time, ratings.iloc[batch]['timestamp'].tolist())
            b_user_edge = torch.from_numpy(b_user_edge).to(device)
            b_users = torch.from_numpy(np.array(ratings.iloc[batch]['user_id'])).to(device) 
            
            b_item_edge = find_latest_1D(np.array(ratings.iloc[batch]['item_id']), adj_item_edge, adj_item_time, ratings.iloc[batch]['timestamp'].tolist())
            b_item_edge = torch.from_numpy(b_item_edge).to(device)
            b_items = torch.from_numpy(np.array(ratings.iloc[batch]['item_id'])).to(device)
            timestamps = torch.from_numpy(np.array(ratings.iloc[batch]['timestamp'])).to(device)
            
            negative_samples = sampler_global(items, adj_user, ratings.iloc[batch]['user_id'].tolist(), len(items)-1, b_items=b_items)
            # 采集了100个负例，这个要换成全局  
            neg_edge = find_latest(negative_samples, adj_item_edge, adj_item_time, ratings.iloc[batch]['timestamp'].tolist())
            negative_samples = torch.from_numpy(np.array(negative_samples)).to(device)
            item_set = torch.cat([b_items.view(-1,1),negative_samples], dim=1) #batch, 101
            timestamps_set = timestamps.unsqueeze(1).repeat(1,len(items))
            neg_edge = torch.from_numpy(neg_edge).to(device)
            edge_set = torch.cat([b_item_edge.view(-1,1),neg_edge], dim=1) #batch, 101
            
            user_embeddings = model(b_users, b_user_edge,timestamps, config.n_layer, nodetype='user')
            
            item_set_list = torch.split(item_set, test_batch_size, dim=1)
            edge_set_list = torch.split(edge_set, test_batch_size, dim=1)
            timestamps_set_list = torch.split(timestamps_set, test_batch_size, dim=1)
            
            itemset_embeddings = []
            for i in range(len(item_set_list)):
                tmp_itemset_embeddings = model(item_set_list[i].flatten(), edge_set_list[i].flatten(), timestamps_set_list[i].flatten(), config.n_layer, nodetype='item')
                itemset_embeddings.append(tmp_itemset_embeddings.view(count, -1, config.embed_dim))
            itemset_embeddings = torch.cat(itemset_embeddings, dim=1)

            # itemset_embeddings = model(item_set.flatten(), edge_set.flatten(), timestamps_set.flatten(), config.n_layer, nodetype='item')
            # itemset_embeddings = itemset_embeddings.view(count, 101, -1)
            
            logits = torch.bmm(user_embeddings.unsqueeze(1), itemset_embeddings.permute(0,2,1)).squeeze(1) # [count,101]
            logits = -logits.cpu().numpy()
            rank = logits.argsort().argsort()[:,0]
            
            recall5 += np.array(rank<5).astype(float).sum()
            recall10 += np.array(rank<10).astype(float).sum()
            NDCG5 += (1 / np.log2(rank + 2))[rank<5].sum()
            NDCG10 += (1 / np.log2(rank + 2))[rank<10].sum()
            
        recall5 = recall5/num_sample
        recall10 = recall10/num_sample
        NDCG5 = NDCG5/num_sample
        NDCG10 = NDCG10/num_sample
            
        print("===> recall_5: {:.10f}, recall_10: {:.10f}, NDCG_5: {:.10f}, NDCG_10: {:.10f}, time:{}".format(recall5, recall10, NDCG5, NDCG10, time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))))

    return recall5, recall10, NDCG5, NDCG10

# liu:12.06-global neg sampler
def sampler_global(items, adj_user, b_users, size, b_items=None):
    negs = []
    for i in range(len(b_users)):      
        houxuan = list(set(items)-set([b_items[i].item()]))
        src_index = random.sample(list(range(len(houxuan))), size)
        negs.append(np.array(houxuan)[src_index])
    negs = np.array(negs)
    return negs


def sampler(items, adj_user, b_users, size):
    negs = []
    for user in b_users:      
        houxuan = list(set(items)-set(adj_user[user]))
        src_index = random.sample(list(range(len(houxuan))), size)
        negs.append(np.array(houxuan)[src_index])
    negs = np.array(negs)
    return negs


def find_latest(nodes, adj, adj_time, timestamps):
    #negative_samples, [b,size]
    edge = np.zeros_like(nodes)
    for ix in range(nodes.shape[0]):
        for iy in range(nodes.shape[1]):
            node = nodes[ix, iy]
            edge_idx = np.searchsorted(adj_time[node], timestamps[ix])-1
            edge[ix, iy] = np.array(adj[node])[edge_idx]
    return edge

def find_latest_1D(nodes, adj, adj_time, timestamps):
    #negative_samples, [b,size]
    edge = np.zeros_like(nodes)
    for ix in range(nodes.shape[0]):
        node = nodes[ix]
        edge_idx = np.searchsorted(adj_time[node], timestamps[ix])-1
        edge[ix] = np.array(adj[node])[edge_idx]
    return edge


if __name__=='__main__':

    config = Config()
    checkpoint_dir='/models'  
    min_NDCG10 = 1000.0
    max_recall10 = 0.0
    max_NDCG10 = 0.0
    # best_epoch = 0
    max_itrs = 0
    
    device_string = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device_string = 'cpu'
    device = torch.device(device_string)

    print("loading the dataset...")
    # ratings, train_data, valid_data, test_data = data_partition('data/ml-1m')
    if config.data_raw:
        ratings, train_data, valid_data, test_data, user_ids_invmap, item_ids_invmap = data_partition(config.data_path)
        with open(os.path.join(config.data_path,'ratings.pkl'),'wb') as f:
            pkl.dump(ratings,f)
        with open(os.path.join(config.data_path,'train_data.pkl'),'wb') as f:
            pkl.dump(train_data,f)
        with open(os.path.join(config.data_path,'valid_data.pkl'),'wb') as f:
            pkl.dump(valid_data,f)
        with open(os.path.join(config.data_path,'test_data.pkl'),'wb') as f:
            pkl.dump(test_data,f)
        with open(os.path.join(config.data_path, 'user_ids_invmap.pkl'), 'wb') as f:
            pkl.dump(user_ids_invmap, f)
        with open(os.path.join(config.data_path, 'item_ids_invmap.pkl'), 'wb') as f:
            pkl.dump(item_ids_invmap, f)
            
    else:
        with open(os.path.join(config.data_path,'ratings.pkl'),'rb') as f:
            ratings = pkl.load(f)
        with open(os.path.join(config.data_path,'train_data.pkl'),'rb') as f:
            train_data = pkl.load(f)
        with open(os.path.join(config.data_path,'valid_data.pkl'),'rb') as f:
            valid_data = pkl.load(f)
        with open(os.path.join(config.data_path,'test_data.pkl'),'rb') as f:
            test_data = pkl.load(f)
        with open(os.path.join(config.data_path, 'user_ids_invmap.pkl'), 'rb') as f:
            user_ids_invmap = pkl.load(f)
        with open(os.path.join(config.data_path, 'item_ids_invmap.pkl'), 'rb') as f:
            item_ids_invmap = pkl.load(f)

    print("Finishing Gen Dataset")
    # exit()
    print(ratings.shape)
    # exit()
    
    users = ratings['user_id'].unique()
    items = ratings['item_id'].unique() 
    items_in_data = ratings.iloc[train_data+valid_data+test_data]['item_id'].unique()

    adj_user = {user: ratings[ratings.user_id == user]['item_id'].tolist() for user in users}
    adj_user_edge = {user:ratings[ratings.user_id == user].index.tolist() for user in users}
    adj_user_time = {user:ratings[ratings.user_id == user]['timestamp'].tolist() for user in users} 
    
    adj_item_edge = {item:ratings[ratings.item_id == item].index.tolist() for item in items}
    adj_item_time = {item:ratings[ratings.item_id == item]['timestamp'].tolist() for item in items} 
    
    num_users = len(users)
    num_items = len(items)
    neighor_finder = NeighborFinder(ratings)
    time_encoder = time_encoding(config.time_dim)
    MLPLayer = MergeLayer(config.embed_dim, config.embed_dim, config.embed_dim, 1)

    a_users = np.array(ratings['user_id'])
    a_items = np.array(ratings['item_id'])
    edge_idx = np.arange(0, len(a_users))

    # 保存邻居，max(config.n_degree)表示要采样的邻居的个数。
    # 只保留每条交互，最近的50个邻居
    user_neig50 = neighor_finder.get_user_neighbor_ind(a_users, edge_idx, max(config.n_degree), device)
    item_neig50 = neighor_finder.get_item_neighbor_ind(a_items, edge_idx, max(config.n_degree), device)
    
    # for revise, 
    # user_neig50, user_egdes50, user_neig_time50, user_neig_mask50 = user_neig50
    # item_neig50, item_egdes50, item_neig_time50, item_neig_mask50 = item_neig50
    # print(user_neig50.shape, item_neig50.shape, user_egdes50.shape, item_egdes50.shape, user_neig_mask50.shape, item_neig_mask50.shape)
    # exit()
    
    ###########################################
    # for discrete dyg
    # ddyg: list, and each element is a subset of ratings with index unchanged
    # ddygs = get_ddygs(ratings, config.num_shots)
    
    # print(len(ddygs))
    # for ddyg in ddygs:
    #     print(ddyg.shape)
    # exit()
    ###########################################
    
    criterion = torch.nn.CrossEntropyLoss(reduction='sum')
    
    model = PTGCN(user_neig50, item_neig50, num_users, num_items,
                 time_encoder, config.n_layer,  config.n_degree, config.node_dim, config.time_dim,
                 config.embed_dim, device, config.n_head, config.drop_out
                 ).to(device)
  
    optim = torch.optim.Adam(model.parameters(),lr=config.lr)

    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    print(num_params)
    
    # 训练集分为不同batch
    dl = DataLoader(train_data, config.batch_size, shuffle=True, pin_memory=True)
    
    itrs = 0
    sum_loss=0
    for epoch in range(config.n_epoch):
        time1 = 0.0
        x=0.0
        for id,batch in enumerate(dl):
            #print('epoch:',epoch,' batch:',id)
            x=x+1
            optim.zero_grad()
            
            count = len(batch)
            
            b_user_edge = find_latest_1D(np.array(ratings.iloc[batch]['user_id']), adj_user_edge, adj_user_time, ratings.iloc[batch]['timestamp'].tolist())
            b_user_edge = torch.from_numpy(b_user_edge).to(device)
            b_users = torch.from_numpy(np.array(ratings.iloc[batch]['user_id'])).to(device) 
            
            b_item_edge = find_latest_1D(np.array(ratings.iloc[batch]['item_id']), adj_item_edge, adj_item_time, ratings.iloc[batch]['timestamp'].tolist())
            b_item_edge = torch.from_numpy(b_item_edge).to(device)
            b_items = torch.from_numpy(np.array(ratings.iloc[batch]['item_id'])).to(device)
            timestamps = torch.from_numpy(np.array(ratings.iloc[batch]['timestamp'])).to(device)
           
            negative_samples = sampler(items_in_data, adj_user, ratings.iloc[batch]['user_id'].tolist() ,1) 
            neg_edge = find_latest(negative_samples, adj_item_edge, adj_item_time, ratings.iloc[batch]['timestamp'].tolist())
            negative_samples = torch.from_numpy(np.array(negative_samples)).to(device)
            negative_samples = negative_samples.squeeze()
            neg_edge = torch.from_numpy(neg_edge).to(device)
            neg_edge = neg_edge.squeeze()

            time0 = time.time()

            user_embeddings = model(b_users, b_user_edge, timestamps, config.n_layer, nodetype='user')
            item_embeddings = model(b_items, b_item_edge, timestamps, config.n_layer, nodetype='item')
            negs_embeddings = model(negative_samples, neg_edge, timestamps, config.n_layer, nodetype='item')
            
            with torch.no_grad():
                labels = torch.zeros(count, dtype=torch.long).to(device)
            l_pos = torch.bmm(user_embeddings.view(count, 1, -1), item_embeddings.view(count, -1, 1)).view(count, 1) # [count,1] 
            
            l_u = torch.bmm(user_embeddings.view(count, 1, -1), negs_embeddings.view(count, -1, 1)).view(count, 1) # [count,n_negs]           
            logits = torch.cat([l_pos, l_u], dim=1)  # [count, 2]
            loss = criterion(logits/config.temperature, labels)

            loss.backward()
            optim.step()
            itrs += 1
            #time1 = time1 + (time.time() - time0)
            #print('time:'+str(time1 / x))

            sum_loss = sum_loss + loss.item()
            avg_loss = sum_loss / itrs 
                   
            if id%10==0:
                print("===>({}/{}, {}): loss: {:.10f}, avg_loss: {:.10f}, time:{}".format(id, len(dl), epoch, loss.item(), avg_loss, time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))))
        
        print("===>({}): loss: {:.10f}, avg_loss: {:.10f}, time:{}".format(epoch, loss.item(), avg_loss, time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))))
             
            
        ### Validation
        # if epoch%5==0:
        if epoch%3==0:
            val_bl = DataLoader(valid_data, 5, shuffle=True, pin_memory=True)
            recall5, recall10, NDCG5, NDCG10 = evaluate_val(model, ratings, items, val_bl, adj_user_edge, adj_item_edge, adj_user_time, adj_item_time, device)
        
        # if recall10 > max_recall10: 
        #     max_recall10 = recall10
        #     max_itrs = 0
        # if min_NDCG10>NDCG10:
        #     min_NDCG10 = NDCG10
        #     max_itrs = 0
        if max_NDCG10 < NDCG10:
            max_NDCG10 = NDCG10
            max_itrs = 0
            torch.save(model.state_dict(), "model_full.pth")
        else:   
            max_itrs += 1
            if max_itrs>config.patience:
                break
            
    # print(f'best_recall@10:{max_recall10}')

    # torch.save(model.state_dict(), "model_full.pth")
    print('Epoch %d test' % epoch)
    model = PTGCN(user_neig50, item_neig50, num_users, num_items, 
                  time_encoder, config.n_layer, config.n_degree, config.node_dim, config.time_dim,
                  config.embed_dim, device, config.n_head, config.drop_out
                  ).to(device)
    model.load_state_dict(torch.load("model_full.pth"))
    model.eval()
    test_bl1 = DataLoader(test_data, 5, shuffle=True, pin_memory=True)
    # recall5, recall10, NDCG5, NDCG10 = evaluate(model, ratings, items, test_bl1, adj_user_edge, adj_item_edge, adj_user_time, adj_item_time, device)
    recall5, recall10, NDCG5, NDCG10 = evaluate_val(model, ratings, items, test_bl1, adj_user_edge, adj_item_edge, adj_user_time, adj_item_time, device)
