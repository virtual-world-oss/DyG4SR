import numpy as np
import pandas as pd
import torch
import os
import random
import csv

data_path = './data/CDs/Amazon_CDs_and_Vinyl.inter'
ratings = pd.read_csv(data_path, sep='\t')
print(ratings)
ratings = ratings.iloc[:,:4]
print(ratings)
ratings.rename(columns={
        'user_id:token': 'user_id',
        'item_id:token': 'item_id',
        'rating:float': 'rating',
        'timestamp:float': 'timestamp'
        }, inplace=True)
df = pd.DataFrame(ratings)

# 将DataFrame转换为二维列表
data = [df.columns.tolist()] + df.values.tolist()

with open('./data/CDs/ratings.dat', 'w', newline='') as f:
    writer = csv.writer(f, delimiter=',')
    writer.writerows(data)

with open('./data/CDs/ratings.dat', 'r') as f:
    content = f.read()
    
# 将','替换为'::'
content = content.replace(',', '::')
# 将替换后的内容写回文件
with open('./data/CDs/ratings.dat', 'w') as f:
    f.write(content)