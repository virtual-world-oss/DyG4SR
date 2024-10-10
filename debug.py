
# import pandas as pd
# import os
# ratings = []
# with open("data/beauty/ratings.dat") as f:
#     for l in f:
#         user_id, item_id, rating, timestamp = [_ for _ in l.split('::')]
#         rating = float(rating)
#         timestamp = int(timestamp)
#         ratings.append({
#                 'user_id': user_id,
#                 'item_id': item_id,
#                 'rating': rating,
#                 'timestamp': timestamp,
#                 })
# ratings = pd.DataFrame(ratings)
# print(ratings.shape)

# from utils import hierarchical_contrastive_loss
# import torch
# batch_size = 64
# embedding_dim = 128
# num_positive = 10
# num_weak_positive = 5
# num_negative = 20
# tau = 0.1  # Temperature

# # Create random embeddings for the batch
# e_i = torch.randn(batch_size, embedding_dim)  # Anchor embeddings, shape (B, D)
# print(e_i.shape)
# P_i = torch.randn(batch_size, num_positive, embedding_dim)  # Positive set, shape (B, P, D)
# print(P_i.shape)
# weak_P_i = torch.randn(batch_size, num_weak_positive, embedding_dim)  # Weak positive set, shape (B, W, D)
# print(weak_P_i.shape)
# N_i = torch.randn(batch_size, num_negative, embedding_dim)  # Negative set, shape (B, N, D)
# print(N_i.shape)

# # Calculate loss
# loss = hierarchical_contrastive_loss(e_i, P_i, N_i, weak_P_i, tau)
# print(loss.shape)  
# print(loss)  # Output the loss values
from torch_geometric.nn import GATConv
import networkx as nx
