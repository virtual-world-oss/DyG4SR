import numpy as np
import pandas as pd
import torch
import os
import random
import pickle as pkl

import torch
import torch.nn.functional as F




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


if __name__ == '__main__':
    # ddygs, time_list = get_ddygs_beauty('data/beauty/ratings.pkl', num_shots=30)
    ddygs, time_list = get_ddygs_beauty_dense('data/beauty/ratings.pkl', num_shots=30)
    # for i, ddyg in enumerate(ddygs):
    #   print(i, ddyg.shape)
    # for i, time in enumerate(time_list):
    #   print(i, time)
    with open('data/beauty/ddygs.pkl', 'wb') as f:
        pkl.dump(ddygs, f)