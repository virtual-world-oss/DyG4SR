import torch
import torch.nn as nn
from modules import TemporalAttentionLayer, TemporalTransformerConv, AttentionFusion, time_encoding, BipartiteGAT
from utils import contrastive_loss, hierarchical_contrastive_loss
from torch_geometric.nn import GATConv

class DyG4SR(nn.Module):
    def __init__(self, user_neig50, item_neig50, ddyg_user_neig50, ddyg_item_neig50, shots_num, ddyg_edges_idx, num_users, num_items, time_encoder, n_layers, n_neighbors,
               n_node_features, n_time_features, embedding_dimension, device,
               n_heads=2, dropout=0.1, config=None):
        super(DyG4SR, self).__init__()
    
        if config == None:
            seed = 2024
        else:
            seed = config.seed
        #################################
        # 设置各个模块的seed
        # random.seed(seed)
        # np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        #################################

        self.num_users = num_users
        self.num_items = num_items
        self.shots_num = shots_num
        self.ddyg_edges_idx = ddyg_edges_idx
        self.time_encoder = time_encoder
        self.n_layers = n_layers
        self.n_time_features = n_time_features
        self.dropout = dropout
        self.embedding_dimension = embedding_dimension
        self.device = device
        self.n_neighbors = n_neighbors
    
        self.user_embeddings = nn.Embedding(self.num_users, self.embedding_dimension)
        self.item_embeddings = nn.Embedding(self.num_items, self.embedding_dimension)
        self.time_embeddings = nn.Embedding(20, self.embedding_dimension)
        nn.init.normal_(self.user_embeddings.weight, std=0.1)
        nn.init.normal_(self.item_embeddings.weight, std=0.1)
        nn.init.normal_(self.time_embeddings.weight, std=0.1)
        
        self.user_neig50, self.user_egdes50, self.user_neig_time50, self.user_neig_mask50 = user_neig50
        self.item_neig50, self.item_egdes50, self.item_neig_time50, self.item_neig_mask50 = item_neig50
        
        self.ddyg_user_neig50, self.ddyg_user_egdes50, self.ddyg_user_neig_time50, self.ddyg_user_neig_mask50 = [list(t) for t in zip(*ddyg_user_neig50)]
        self.ddyg_item_neig50, self.ddyg_item_egdes50, self.ddyg_item_neig_time50, self.ddyg_item_neig_mask50 = [list(t) for t in zip(*ddyg_item_neig50)]

        self.shotsfusion = TemporalTransformerConv(self.embedding_dimension)
        self.globallocalfusion = AttentionFusion(self.embedding_dimension)
        # self.node_time_encoder = time_encoding(self.embedding_dimension)
        
        self.attention_models = torch.nn.ModuleList([TemporalAttentionLayer(
            n_node_features=n_node_features,
            n_neighbors_features=n_node_features,
            time_dim=n_time_features,
            output_dimension=embedding_dimension,   
            n_head=n_heads,
            n_neighbor= self.n_neighbors[i],
            dropout=dropout)
            for i in range(n_layers)])
        
        ###
        #for pre train
        self.pretrain_GNN = BipartiteGAT(self.embedding_dimension, self.embedding_dimension, self.embedding_dimension, dropout=self.dropout)

    def compute_embedding(self, nodes, edges, timestamps, n_layers, nodetype='user'):
        """
        src_idx_l [batch_size]: users / items input ids.
        cut_time_l [batch_size]: scalar representing the instant of the time where we want to extract the user / item representation.
        curr_layers [scalar]: number of temporal convolutional layers to stack.
        num_neighbors [scalar]: number of temporal neighbor to consider in each convolutional layer.
        """
        #assert (n_layers >= 0)
        device = nodes.device

        n_neighbor = self.n_neighbors[n_layers-1]
        nodes_torch = nodes.long()
        edges_torch = edges.long()
        timestamps_torch = timestamps.long()
        
        #inx = torch.arange(0,20).to(device)

        # query node always has the start time -> time span == 0
        #nodes_time_embedding = torch.matmul(self.time_encoder(torch.zeros_like(timestamps_torch)),self.time_embeddings(inx)).unsqueeze(1)
        nodes_time_embedding = self.time_embeddings(self.time_encoder(torch.zeros_like(timestamps_torch)))
        # nodes_time_embedding = self.time_embeddings(self.node_time_encoder(torch.zeros_like(timestamps_torch)))
        # nodes_time_embedding = self.time_embeddings(torch.zeros_like(timestamps_torch).long())
        if nodetype=='user':
            node_features = self.user_embeddings(nodes_torch)
        else:
            node_features = self.item_embeddings(nodes_torch)
            
        if n_layers == 0:
            return node_features
        else:
            if nodetype=='user':
                
                adj, adge, times, mask = self.user_neig50[edges_torch,-n_neighbor:], self.user_egdes50[edges_torch,-n_neighbor:], self.user_neig_time50[edges_torch,-n_neighbor:], self.user_neig_mask50[edges_torch,-n_neighbor:]
                
                edge_deltas = timestamps_torch.unsqueeze(1) - times   #[batch_size,n_neighors]
                # edge_deltas = times
                adj = adj.flatten()
                times = times.flatten()
                adge = adge.flatten()

                neighbor_embeddings = self.compute_embedding(adj, adge, times, n_layers - 1, 'item')
                neighbor_embeddings = neighbor_embeddings.view(len(nodes), n_neighbor, -1)
                # origin
                edge_time_embeddings = self.time_embeddings(self.time_encoder(edge_deltas.flatten()))
                # alter
                # edge_time_embeddings = self.time_encoder(edge_deltas.flatten())
                #edge_time_embeddings = torch.matmul(self.time_encoder(edge_deltas.flatten()),self.time_embeddings(inx))
                edge_time_embeddings = edge_time_embeddings.view(len(nodes), n_neighbor, -1)

                node_embedding,_  = self.attention_models[n_layers - 1](node_features,
                                        nodes_time_embedding,
                                        neighbor_embeddings,
                                        edge_time_embeddings,
                                        mask)
            
        
            if nodetype=='item':
                adj, adge, times, mask = self.item_neig50[edges_torch,-n_neighbor:], self.item_egdes50[edges_torch,-n_neighbor:], self.item_neig_time50[edges_torch,-n_neighbor:], self.item_neig_mask50[edges_torch,-n_neighbor:]
                
                edge_deltas = timestamps_torch.unsqueeze(1) - times   #[batch_size,n_neighors]
                # edge_deltas = times
                adj = adj.flatten()
                times = times.flatten()
                adge = adge.flatten()
                
                neighbor_embeddings = self.compute_embedding(adj, adge, times, n_layers - 1, 'user')
                neighbor_embeddings = neighbor_embeddings.view(len(nodes), n_neighbor, -1)
                # origin
                edge_time_embeddings = self.time_embeddings(self.time_encoder(edge_deltas.flatten()))
                # alter
                # edge_time_embeddings = self.time_encoder(edge_deltas.flatten())
                
                #edge_time_embeddings = torch.matmul(self.time_encoder(edge_deltas.flatten()),self.time_embeddings(inx))
                edge_time_embeddings = edge_time_embeddings.view(len(nodes), n_neighbor, -1)

                node_embedding,_ = self.attention_models[n_layers - 1](node_features,
                                        nodes_time_embedding,
                                        neighbor_embeddings,
                                        edge_time_embeddings,
                                        mask)
        return node_embedding
    
    
    def compute_ddyg_embedding(self, nodes, edges, timestamps, n_layers, nodetype='user'):
        ddyg_node_embedding = []
        for idx in range(self.shots_num):
            ddyg_node_embedding.append(self.compute_ddyg_embedding_oneshots(nodes, edges, timestamps, n_layers, nodetype, idx))
        cl_losses = []
        for idx in range(self.shots_num):
            if idx >= self.shots_num-1:
                break
            cl_losses.append(contrastive_loss(ddyg_node_embedding[idx], ddyg_node_embedding[(idx+1)]))
        if len(cl_losses) > 0:
            cl_loss = torch.mean(torch.stack(cl_losses))
        ddyg_node_embedding = torch.stack(ddyg_node_embedding, dim=0)
        ddyg_node_embedding = self.shotsfusion(ddyg_node_embedding).squeeze(0)
        # print(ddyg_node_embedding.shape)
        # exit()
        return ddyg_node_embedding, cl_loss
    
    def compute_ddyg_embedding_oneshots(self, nodes, edges, timestamps, n_layers, nodetype='user', shot_idx=None):
        """
        src_idx_l [batch_size]: users / items input ids.
        cut_time_l [batch_size]: scalar representing the instant of the time where we want to extract the user / item representation.
        curr_layers [scalar]: number of temporal convolutional layers to stack.
        num_neighbors [scalar]: number of temporal neighbor to consider in each convolutional layer.
        """
        #assert (n_layers >= 0)
        device = nodes.device

        n_neighbor = self.n_neighbors[n_layers-1]
        nodes_torch = nodes.long()
        edges_torch = edges.long()
        timestamps_torch = timestamps.long()
        
        #inx = torch.arange(0,20).to(device)

        # query node always has the start time -> time span == 0
        #nodes_time_embedding = torch.matmul(self.time_encoder(torch.zeros_like(timestamps_torch)),self.time_embeddings(inx)).unsqueeze(1)
        # print('nodes_time_embedding generating')
        # print(timestamps_torch.shape)
        nodes_time_embedding = self.time_embeddings(self.time_encoder(torch.zeros_like(timestamps_torch)))
        # nodes_time_embedding = self.time_embeddings(self.node_time_encoder(torch.zeros_like(timestamps_torch)))
        # nodes_time_embedding = self.time_embeddings(torch.zeros_like(timestamps_torch).long())
        if nodetype=='user':
            node_features = self.user_embeddings(nodes_torch)
        else:
            node_features = self.item_embeddings(nodes_torch)
            
        if n_layers == 0:
            return node_features
        else:
            ddyg_edge_idx = self.ddyg_edges_idx[shot_idx]
            # print(f'min: {ddyg_edge_idx[0]}, max: {ddyg_edge_idx[1]}')
            # if shot_idx>=1:
            #     print(f'before calculating; max edge:{torch.max(edges_torch)}, min edge:{torch.min(edges_torch)}')
            valid_mask = (edges_torch >= ddyg_edge_idx[0]) & (edges_torch <= ddyg_edge_idx[1])
            invalid_mask = ~valid_mask
            edges_torch = edges_torch[valid_mask]
            if len(edges_torch)==0:
                return node_features
            if not torch.min(edges_torch)-ddyg_edge_idx[0] < 0:
                edges_torch = edges_torch - ddyg_edge_idx[0]
            nodes_torch = nodes_torch[valid_mask]
            timestamps_torch = timestamps_torch[valid_mask]
            node_features_vlid = node_features[valid_mask]
            nodes_time_embedding = nodes_time_embedding[valid_mask]
            
            if nodetype=='user':
                
                # adj, adge, times, mask = self.user_neig50[edges_torch,-n_neighbor:], self.user_egdes50[edges_torch,-n_neighbor:], self.user_neig_time50[edges_torch,-n_neighbor:], self.user_neig_mask50[edges_torch,-n_neighbor:]
                # if shot_idx>=1:
                #     print(f'max edge:{torch.max(edges_torch)}, min edge:{torch.min(edges_torch)}')
                
                adj, adge, times, mask = self.ddyg_user_neig50[shot_idx][edges_torch,-n_neighbor:], self.ddyg_user_egdes50[shot_idx][edges_torch,-n_neighbor:], self.ddyg_user_neig_time50[shot_idx][edges_torch,-n_neighbor:], self.ddyg_user_neig_mask50[shot_idx][edges_torch,-n_neighbor:]
                # if shot_idx>=1:
                #     print(f'edge shape:{adge.shape}')
                if torch.min(adge) < ddyg_edge_idx[0]:
                    adge = adge + ddyg_edge_idx[0]
                
                edge_deltas = timestamps_torch.unsqueeze(1) - times   #[batch_size,n_neighors]
                # edge_deltas = times
                adj = adj.flatten()
                times = times.flatten()
                adge = adge.flatten()

                # print('here user info generating')
                neighbor_embeddings = self.compute_ddyg_embedding_oneshots(adj, adge, times, n_layers - 1, 'item',shot_idx=shot_idx)
                # print('user exit')
                neighbor_embeddings = neighbor_embeddings.view(len(nodes_torch), n_neighbor, -1)
                # origin
                edge_time_embeddings = self.time_embeddings(self.time_encoder(edge_deltas.flatten()))
                # alter
                # edge_time_embeddings = self.time_encoder(edge_deltas.flatten())
                #edge_time_embeddings = torch.matmul(self.time_encoder(edge_deltas.flatten()),self.time_embeddings(inx))
                edge_time_embeddings = edge_time_embeddings.view(len(nodes_torch), n_neighbor, -1)

                # print('here user input attention')
                node_embedding,_  = self.attention_models[n_layers - 1](node_features_vlid,
                                        nodes_time_embedding,
                                        neighbor_embeddings,
                                        edge_time_embeddings,
                                        mask)
                # print('user exit attention')
                
            if nodetype=='item':
                # adj, adge, times, mask = self.item_neig50[edges_torch,-n_neighbor:], self.item_egdes50[edges_torch,-n_neighbor:], self.item_neig_time50[edges_torch,-n_neighbor:], self.item_neig_mask50[edges_torch,-n_neighbor:]
                # if shot_idx>=1:
                    # print(f'max edge:{torch.max(edges_torch)}, min edge:{torch.min(edges_torch)}')
                adj, adge, times, mask = self.ddyg_item_neig50[shot_idx][edges_torch,-n_neighbor:], self.ddyg_item_egdes50[shot_idx][edges_torch,-n_neighbor:], self.ddyg_item_neig_time50[shot_idx][edges_torch,-n_neighbor:], self.ddyg_item_neig_mask50[shot_idx][edges_torch,-n_neighbor:]
                # if shot_idx>=1:
                #     print(f'edge shape:{adge.shape}')
                if torch.min(adge) < ddyg_edge_idx[0]:
                    adge = adge + ddyg_edge_idx[0]
                
                edge_deltas = timestamps_torch.unsqueeze(1) - times   #[batch_size,n_neighors]
                # edge_deltas = times
                adj = adj.flatten()
                times = times.flatten()
                adge = adge.flatten()
                
                # print('here item info generating')
                neighbor_embeddings = self.compute_ddyg_embedding_oneshots(adj, adge, times, n_layers - 1, 'user',shot_idx=shot_idx)
                # print('item exit')
                neighbor_embeddings = neighbor_embeddings.view(len(nodes_torch), n_neighbor, -1)
                # origin
                edge_time_embeddings = self.time_embeddings(self.time_encoder(edge_deltas.flatten()))
                # alter
                # edge_time_embeddings = self.time_encoder(edge_deltas.flatten())
                #edge_time_embeddings = torch.matmul(self.time_encoder(edge_deltas.flatten()),self.time_embeddings(inx))
                edge_time_embeddings = edge_time_embeddings.view(len(nodes_torch), n_neighbor, -1)

                # print('here item input attention')
                node_embedding,_ = self.attention_models[n_layers - 1](node_features_vlid,
                                        nodes_time_embedding,
                                        neighbor_embeddings,
                                        edge_time_embeddings,
                                        mask)
                # print('item attention exit')
            
            node_features[valid_mask] = node_embedding
            node_embedding = node_features
        return node_embedding

    def pre_train_embedding(self, anchor_item, poitive_item_neighbors, weak_popular_items, negative_item_neighbors, pre_train_edges):
        user_embeddings = self.user_embeddings.weight
        item_embeddings = self.item_embeddings.weight
        all_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        pre_train_embeddings = self.pretrain_GNN(all_embeddings, pre_train_edges)
        anchor_item_embedding = pre_train_embeddings[anchor_item]
        poitive_item_neighbors_embedding = pre_train_embeddings[poitive_item_neighbors]
        weak_popular_items_embedding = pre_train_embeddings[weak_popular_items]
        negative_item_neighbors_embedding = pre_train_embeddings[negative_item_neighbors]
        
        # print(anchor_item_embedding.shape)
        # print(poitive_item_neighbors_embedding.shape)
        # print(negative_item_neighbors_embedding.shape)
        # print(weak_popular_items_embedding.shape)
        # exit()
        return hierarchical_contrastive_loss(anchor_item_embedding, poitive_item_neighbors_embedding, negative_item_neighbors_embedding, weak_popular_items_embedding)

    def forward(self, nodes, edges, timestamps, n_layers, nodetype='user'):
        # global_node_embedding = self.compute_embedding(nodes, edges, timestamps, n_layers, nodetype)
        local_node_embedding, cl_loss = self.compute_ddyg_embedding(nodes, edges, timestamps, n_layers, nodetype)
        # node_embedding = torch.mean(global_node_embedding, local_node_embedding, dim=1)
        # node_embedding = (global_node_embedding + local_node_embedding) / 2
        # node_embedding = self.globallocalfusion(global_node_embedding, local_node_embedding)
        node_embedding = local_node_embedding
        return node_embedding, cl_loss
        



class PTGCN(nn.Module):
    def __init__(self, user_neig50, item_neig50, num_users, num_items, time_encoder, n_layers, n_neighbors,
               n_node_features, n_time_features, embedding_dimension, device,
               n_heads=2, dropout=0.1):
        super(PTGCN, self).__init__()
    
        self.num_users = num_users
        self.num_items = num_items
        self.time_encoder = time_encoder
        self.n_layers = n_layers
        self.n_time_features = n_time_features
        self.dropout = dropout
        self.embedding_dimension = embedding_dimension
        self.device = device
        self.n_neighbors = n_neighbors
    
        self.user_embeddings = nn.Embedding(self.num_users, self.embedding_dimension)
        self.item_embeddings = nn.Embedding(self.num_items, self.embedding_dimension)
        self.time_embeddings = nn.Embedding(20, self.embedding_dimension)
        nn.init.normal_(self.user_embeddings.weight, std=0.1)
        nn.init.normal_(self.item_embeddings.weight, std=0.1)
        nn.init.normal_(self.time_embeddings.weight, std=0.1)
        
        self.user_neig50, self.user_egdes50, self.user_neig_time50, self.user_neig_mask50 = user_neig50
        self.item_neig50, self.item_egdes50, self.item_neig_time50, self.item_neig_mask50 = item_neig50
    
        self.attention_models = torch.nn.ModuleList([TemporalAttentionLayer(
            n_node_features=n_node_features,
            n_neighbors_features=n_node_features,
            time_dim=n_time_features,
            output_dimension=embedding_dimension,   
            n_head=n_heads,
            n_neighbor= self.n_neighbors[i],
            dropout=dropout)
            for i in range(n_layers)])

    def compute_embedding(self, nodes, edges, timestamps, n_layers, nodetype='user'):
        """
        src_idx_l [batch_size]: users / items input ids.
        cut_time_l [batch_size]: scalar representing the instant of the time where we want to extract the user / item representation.
        curr_layers [scalar]: number of temporal convolutional layers to stack.
        num_neighbors [scalar]: number of temporal neighbor to consider in each convolutional layer.
        """
        #assert (n_layers >= 0)
        device = nodes.device

        n_neighbor = self.n_neighbors[n_layers-1]
        nodes_torch = nodes.long()
        edges_torch = edges.long()
        timestamps_torch = timestamps.long()
        
        #inx = torch.arange(0,20).to(device)

        # query node always has the start time -> time span == 0
        #nodes_time_embedding = torch.matmul(self.time_encoder(torch.zeros_like(timestamps_torch)),self.time_embeddings(inx)).unsqueeze(1)
        nodes_time_embedding = self.time_embeddings(self.time_encoder(torch.zeros_like(timestamps_torch)))
        if nodetype=='user':
            node_features = self.user_embeddings(nodes_torch)
        else:
            node_features = self.item_embeddings(nodes_torch)
            
        if n_layers == 0:
            return node_features
        else:
            if nodetype=='user':
                
                adj, adge, times, mask = self.user_neig50[edges_torch,-n_neighbor:], self.user_egdes50[edges_torch,-n_neighbor:], self.user_neig_time50[edges_torch,-n_neighbor:], self.user_neig_mask50[edges_torch,-n_neighbor:]
                
                edge_deltas = timestamps_torch.unsqueeze(1) - times   #[batch_size,n_neighors]
                adj = adj.flatten()
                times = times.flatten()
                adge = adge.flatten()

                neighbor_embeddings = self.compute_embedding(adj, adge, times, n_layers - 1, 'item')
                neighbor_embeddings = neighbor_embeddings.view(len(nodes), n_neighbor, -1)
                edge_time_embeddings = self.time_embeddings(self.time_encoder(edge_deltas.flatten()))
                #edge_time_embeddings = torch.matmul(self.time_encoder(edge_deltas.flatten()),self.time_embeddings(inx))
                edge_time_embeddings = edge_time_embeddings.view(len(nodes), n_neighbor, -1)

                node_embedding,_  = self.attention_models[n_layers - 1](node_features,
                                        nodes_time_embedding,
                                        neighbor_embeddings,
                                        edge_time_embeddings,
                                        mask)
            
        
            if nodetype=='item':
                adj, adge, times, mask = self.item_neig50[edges_torch,-n_neighbor:], self.item_egdes50[edges_torch,-n_neighbor:], self.item_neig_time50[edges_torch,-n_neighbor:], self.item_neig_mask50[edges_torch,-n_neighbor:]
                
                edge_deltas = timestamps_torch.unsqueeze(1) - times   #[batch_size,n_neighors]
                adj = adj.flatten()
                times = times.flatten()
                adge = adge.flatten()
                
                neighbor_embeddings = self.compute_embedding(adj, adge, times, n_layers - 1, 'user')
                neighbor_embeddings = neighbor_embeddings.view(len(nodes), n_neighbor, -1)
                edge_time_embeddings = self.time_embeddings(self.time_encoder(edge_deltas.flatten()))
                #edge_time_embeddings = torch.matmul(self.time_encoder(edge_deltas.flatten()),self.time_embeddings(inx))
                edge_time_embeddings = edge_time_embeddings.view(len(nodes), n_neighbor, -1)

                node_embedding,_ = self.attention_models[n_layers - 1](node_features,
                                        nodes_time_embedding,
                                        neighbor_embeddings,
                                        edge_time_embeddings,
                                        mask)
            
        
        return node_embedding
    
    
    def forward(self, nodes, edges, timestamps, n_layers, nodetype='user'):
        
        return self.compute_embedding(nodes, edges, timestamps, n_layers, nodetype)
        