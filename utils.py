import numpy as np
import pandas as pd
import torch
import os
import random
import pickle as pkl

import torch
import torch.nn.functional as F

# def hierarchical_contrastive_loss(e_i, P_i, N_i, weak_P_i, tau):
#     """
#     Hierarchical contrastive loss with a separate weak positive set.

#     Args:
#         e_i (torch.Tensor): Anchor embeddings, shape (B, D)
#         P_i (torch.Tensor): Positive set embeddings, shape (B, P, D)
#         weak_P_i (torch.Tensor): Weak positive set embeddings, shape (B, W, D)
#         N_i (torch.Tensor): Negative set embeddings, shape (B, N, D)
#         tau (float): Temperature parameter
    
#     Returns:
#         loss (torch.Tensor): Computed hierarchical contrastive loss, shape (B,)
#     """
#     # Compute dot products with positive, weak positive, and negative examples
#     e_i_P_i = torch.matmul(e_i.unsqueeze(1), P_i.transpose(1, 2)) / tau  # Shape (B, 1, P)
#     e_i_weak_P_i = torch.matmul(e_i.unsqueeze(1), weak_P_i.transpose(1, 2)) / tau  # Shape (B, 1, W)
#     e_i_N_i = torch.matmul(e_i.unsqueeze(1), N_i.transpose(1, 2)) / tau  # Shape (B, 1, N)
    
#     # Squeeze to make sure the shapes are correct
#     e_i_P_i = e_i_P_i.squeeze(1)  # Shape (B, P)
#     e_i_weak_P_i = e_i_weak_P_i.squeeze(1)  # Shape (B, W)
#     e_i_N_i = e_i_N_i.squeeze(1)  # Shape (B, N)
    
#     # Apply clamp to prevent overflow
#     e_i_P_i = torch.clamp(e_i_P_i, max=709)  # Clamp to avoid overflow in exp
#     e_i_weak_P_i = torch.clamp(e_i_weak_P_i, max=709)  # Clamp to avoid overflow in exp
#     e_i_N_i = torch.clamp(e_i_N_i, max=709)  # Clamp to avoid overflow in exp
    
#     # First term: log of positive over sum of all positives + negatives + weak positives
#     numerator_1 = torch.exp(e_i_P_i).sum(dim=1)  # Shape (B,)
#     denominator_1 = numerator_1 + torch.exp(e_i_N_i).sum(dim=1) + torch.exp(e_i_weak_P_i).sum(dim=1)  # Shape (B,)
#     print(torch.exp(e_i_P_i))
    
#     # Apply clamp to denominator to prevent division by zero
#     denominator_1 = torch.clamp(denominator_1, min=1e-8)  # Prevent division by zero
#     loss_1 = -torch.log(numerator_1 / denominator_1)
    
#     # Second term: log of weak positive min(M_i) and positive over sum of all positives + negatives + weak positives
#     M_i = e_i_P_i.min(dim=1).values  # Shape (B,) 正例集合中的最小相似度
    
#     # Apply clamp to M_i to avoid overflow
#     M_i = torch.clamp(M_i, max=709)  # Clamp to avoid overflow in exp
#     min_term = torch.min(M_i.unsqueeze(1), e_i_weak_P_i)  # Shape (B, W)
    
#     numerator_2 = torch.exp(min_term).sum(dim=1)  # Shape (B,)
#     denominator_2 = torch.exp(e_i_P_i).sum(dim=1) + torch.exp(e_i_N_i).sum(dim=1) + torch.exp(e_i_weak_P_i).sum(dim=1)  # Shape (B,)
    
#     # Apply clamp to denominator to prevent division by zero
#     denominator_2 = torch.clamp(denominator_2, min=1e-8)  # Prevent division by zero
#     loss_2 = -torch.log(numerator_2 / denominator_2)
    
#     # Total loss is the sum of both terms
#     total_loss = loss_1 + loss_2
#     return total_loss



def hierarchical_contrastive_loss(e_i, P_i, N_i, weak_P_i, tau=0.1):
    """
    Hierarchical contrastive loss with a separate weak positive set.

    Args:
        e_i (torch.Tensor): Anchor embeddings, shape (B, D)
        P_i (torch.Tensor): Positive set embeddings, shape (B, P, D)
        weak_P_i (torch.Tensor): Weak positive set embeddings, shape (B, W, D)
        N_i (torch.Tensor): Negative set embeddings, shape (B, N, D)
        tau (float): Temperature parameter
    
    Returns:
        loss (torch.Tensor): Computed hierarchical contrastive loss, shape (B,)
    """
    # Compute dot products with positive, weak positive, and negative examples
    e_i_P_i = torch.matmul(e_i.unsqueeze(1), P_i.transpose(1, 2)) / tau  # Shape (B, 1, P)
    e_i_weak_P_i = torch.matmul(e_i.unsqueeze(1), weak_P_i.transpose(1, 2)) / tau  # Shape (B, 1, W)
    e_i_N_i = torch.matmul(e_i.unsqueeze(1), N_i.transpose(1, 2)) / tau  # Shape (B, 1, N)
    # print(e_i_P_i)
    # print(e_i_weak_P_i)
    # print(e_i_N_i)
    
    # Squeeze to make sure the shapes are correct
    e_i_P_i = e_i_P_i.squeeze(1)  # Shape (B, P)
    e_i_weak_P_i = e_i_weak_P_i.squeeze(1)  # Shape (B, W)
    e_i_N_i = e_i_N_i.squeeze(1)  # Shape (B, N)
    
    # First term: log of positive over sum of all positives + negatives + weak positives
    numerator_1 = torch.exp(e_i_P_i).sum(dim=1)  # Shape (B,)
    denominator_1 = numerator_1 + torch.exp(e_i_N_i).sum(dim=1) + torch.exp(e_i_weak_P_i).sum(dim=1)  # Shape (B,)
    # print(torch.exp(e_i_P_i))
    loss_1 = -torch.log(numerator_1 / denominator_1)
    # print(loss_1)
    
    # Second term: log of weak positive min(M_i) and positive over sum of all positives + negatives + weak positives
    M_i = e_i_P_i.min(dim=1).values  # Shape (B,) 正例集合中的最小相似度
    min_term = torch.min(M_i.unsqueeze(1), e_i_weak_P_i)  # Shape (B, W)
    numerator_2 = torch.exp(min_term).sum(dim=1)  # Shape (B,)
    denominator_2 = torch.exp(e_i_P_i).sum(dim=1) + torch.exp(e_i_N_i).sum(dim=1) + torch.exp(e_i_weak_P_i).sum(dim=1)  # Shape (B,)
    loss_2 = -torch.log(numerator_2 / denominator_2)
    
    # Total loss is the sum of both terms
    total_loss = loss_1 + loss_2
    return total_loss.mean()
    


def contrastive_loss(X, Y):
    """
    计算两个 BxD 矩阵的 InfoNCE 对比学习损失。
    
    Args:
        X (torch.Tensor): 第一个形状为 (B, D) 的tensor, 表示 B 个样本的 embedding。
        Y (torch.Tensor): 第二个形状为 (B, D) 的tensor, 表示 B 个样本的 embedding。
        
    Returns:
        loss (torch.Tensor): InfoNCE 损失。
    """
    B, D = X.shape

    # 1. 计算正例相似度 (同一行)
    # positive_sim = F.cosine_similarity(X, Y)  # 返回形状为 (B,)
    # 2. 计算负例相似度 (不同行)
    similarity_matrix = torch.mm(X, Y.t())  # 形状为 (B, B)
    # 3. 构建 InfoNCE 损失
    labels = torch.arange(B).to(X.device)  # 正样本的索引，形状为 (B,)

    # 4. 使用 Cross Entropy 损失计算
    loss = F.cross_entropy(similarity_matrix, labels)

    return loss



def get_ddygs_beauty(ratings_path, num_shots):
    with open(ratings_path, 'rb') as f:
        ratings = pkl.load(f)
    times = ratings['timestamp'].tolist()
    MAX_TIME = max(times)
    MIN_TIME = min(times)
    print(MAX_TIME,MIN_TIME)
    TIME_INTERVAL = (MAX_TIME-MIN_TIME)
    print(TIME_INTERVAL)
    NUM_SHOTS = num_shots
    
    ratings_shot = []
    time_list = []
    step_size = np.around(1 / NUM_SHOTS,4)
    for i in np.around(np.arange(0,1,step_size),4):
        down_line = np.around(i, 4) 
        up_line = np.around(i+step_size,4)
        down_line = int(TIME_INTERVAL * down_line + MIN_TIME)
        up_line = int(TIME_INTERVAL * up_line + MIN_TIME)
        print(down_line,up_line)
        now_train_ratings = ratings[(ratings['timestamp'] >= down_line) & (ratings['timestamp'] < up_line)]
        ratings_shot.append(now_train_ratings)
        time_list.append((down_line,up_line))
    return ratings_shot, time_list

def get_ddygs_beauty_dense(ratings_path, num_shots):
    with open(ratings_path, 'rb') as f:
        ratings = pkl.load(f)
    times = ratings['timestamp'].tolist()
    MAX_TIME = max(times)
    MIN_TIME = min(times)
    print(MAX_TIME,MIN_TIME)
    TIME_INTERVAL = (MAX_TIME-MIN_TIME)
    print(TIME_INTERVAL)
    NUM_SHOTS = num_shots
    
    # time_intervals = [(115084800, 446022650),(446022650, 458751029),(458751029, 471479408),(471479408, 484207787),(484207787, 497318400)]
    # 465115218
    time_intervals = [(115084800, 465115218),(465115218, 497318400)]
    ratings_shot = []
    time_list = []
    for interval in time_intervals:
        down_line, up_line = interval
        print(down_line,up_line)
        now_train_ratings = ratings[(ratings['timestamp'] >= down_line) & (ratings['timestamp'] < up_line)]
        ratings_shot.append(now_train_ratings)
        time_list.append((down_line,up_line))
    return ratings_shot, time_list

def get_ddygs_ml_1m(ratings_path, num_shots):
    with open(ratings_path, 'rb') as f:
        ratings = pkl.load(f)
    times = ratings['timestamp'].tolist()
    MAX_TIME = max(times)
    MIN_TIME = min(times)
    print(MAX_TIME,MIN_TIME)
    TIME_INTERVAL = (MAX_TIME-MIN_TIME)
    print(TIME_INTERVAL)
    NUM_SHOTS = num_shots
    
    ratings_shot = []
    time_list = []
    step_size = np.around(1 / NUM_SHOTS,4)
    for i in np.around(np.arange(0,1,step_size),4):
        down_line = np.around(i, 4) 
        up_line = np.around(i+step_size,4)
        down_line = int(TIME_INTERVAL * down_line + MIN_TIME)
        up_line = int(TIME_INTERVAL * up_line + MIN_TIME)
        print(down_line,up_line)
        now_train_ratings = ratings[(ratings['timestamp'] >= down_line) & (ratings['timestamp'] < up_line)]
        ratings_shot.append(now_train_ratings)
        time_list.append((down_line,up_line))
    return ratings_shot, time_list

def get_ddygs_ml_1m_dense(ratings_path, num_shots):
    with open(ratings_path, 'rb') as f:
        ratings = pkl.load(f)
    times = ratings['timestamp'].tolist()
    MAX_TIME = max(times)
    MIN_TIME = min(times)
    print(MAX_TIME,MIN_TIME)
    TIME_INTERVAL = (MAX_TIME-MIN_TIME)
    print(TIME_INTERVAL)
    NUM_SHOTS = num_shots
    
    # time_intervals = [(115084800, 446022650),(446022650, 458751029),(458751029, 471479408),(471479408, 484207787),(484207787, 497318400)]
    # 465115218
    time_intervals = [(0, 17950131),(17950131, 89750658)]
    ratings_shot = []
    time_list = []
    for interval in time_intervals:
        down_line, up_line = interval
        print(down_line,up_line)
        now_train_ratings = ratings[(ratings['timestamp'] >= down_line) & (ratings['timestamp'] < up_line)]
        ratings_shot.append(now_train_ratings)
        time_list.append((down_line,up_line))
    return ratings_shot, time_list


if __name__ == '__main__':
    # ddygs, time_list = get_ddygs_beauty('data/beauty/ratings.pkl', num_shots=30)
    # ddygs, time_list = get_ddygs_beauty_dense('data/beauty/ratings.pkl', num_shots=30)
    # ddygs, time_list = get_ddygs_ml_1m('data/ml-1m/ratings.pkl', num_shots=20)
    ddygs, time_list = get_ddygs_ml_1m_dense('data/ml-1m/ratings.pkl', num_shots=30)
    for i, ddyg in enumerate(ddygs):
      print(i, ddyg.shape)
    for i, time in enumerate(time_list):
      print(i, time)
    with open('data/ml-1m/ddygs.pkl', 'wb') as f:
        pkl.dump(ddygs, f)