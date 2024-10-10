import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.init as init
import torch.nn.functional as F
from torch_geometric.nn.inits import glorot
from torch.nn import Parameter
from torch_geometric.nn import GATConv

class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, max_length=2000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_length, dim_model, requires_grad=False)
        position = torch.arange(0, max_length).unsqueeze(1).float()
        exp_term = torch.exp(torch.arange(0, dim_model, 2).float() * -(math.log(10000.0) / dim_model))
        pe[:, 0::2] = torch.sin(position * exp_term) # take the odd (jump by 2)
        pe[:, 1::2] = torch.cos(position * exp_term) # take the even (jump by 2)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, maxlen):
        """
        args:
            input: B x T x D
        output:
            tensor: B x T
        """
        return self.pe[:, :maxlen]
    
    
# class TimeEncodeco(torch.nn.Module):
#     def __init__(self, expand_dim, factor=5):
#         super(TimeEncodeco, self).__init__()
#         #init_len = np.array([1e8**(i/(time_dim-1)) for i in range(time_dim)])
        
#         time_dim = expand_dim
#         self.factor = factor
#         self.basis_freq = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, time_dim))).float())
#         self.phase = torch.nn.Parameter(torch.zeros(time_dim).float())
        
#         #self.dense = torch.nn.Linear(time_dim, expand_dim, bias=False)

#         #torch.nn.init.xavier_normal_(self.dense.weight)
        
#     def forward(self, ts):
#         # ts: [N, L]
#         batch_size = ts.size(0)
#         seq_len = ts.size(1)
                
#         ts = ts.view(batch_size, seq_len, 1)# [N, L, 1]
#         map_ts = ts * self.basis_freq.view(1, 1, -1) # [N, L, time_dim]
#         map_ts += self.phase.view(1, 1, -1)
        
#         harmonic = torch.cos(map_ts)

#         return harmonic #self.dense(harmonic)

class TimeEncodeco(torch.nn.Module):
    def __init__(self, expand_dim, factor=5):
        super(TimeEncodeco, self).__init__()
        
        time_dim = expand_dim
        self.factor = factor
        self.basis_freq = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, time_dim))).float())
        self.phase = torch.nn.Parameter(torch.zeros(time_dim).float())
        
    def forward(self, ts):
        # ts: [N], 输入是 [N] 而不是 [N, L]
        batch_size = ts.size(0)
        
        # 将 [N] 转换为 [N, 1]，以便后续广播
        ts = ts.view(batch_size, 1)  # [N, 1]

        # 计算时间编码，将其扩展为 [N, time_dim]
        map_ts = ts * self.basis_freq.view(1, -1)  # [N, time_dim]
        map_ts += self.phase.view(1, -1)
        
        # 计算余弦编码
        harmonic = torch.cos(map_ts)

        return harmonic  # 返回的是 [N, time_dim]


class TimeEncode(torch.nn.Module):
  # Time Encoding proposed by TGAT
  def __init__(self, H):
    super(TimeEncode, self).__init__()

    self.H = H
    self.w = torch.nn.Linear(1, H)
    self.em = torch.nn.Linear(H, H)
    self.act = torch.nn.LeakyReLU()
    self.soft = nn.Softmax(dim=1)
    
  def forward(self, t):
    # t has shape [batch_size, seq_len]
    # Add dimension at the end to apply linear layer --> [batch_size, seq_len, 1]
    t = t.unsqueeze(1)
    output1 = self.w(t.float())
    output2 = self.act(output1)
    output = self.em(output2) + output2
    output = self.soft(output)
    #output = output.unsqueeze(1)
    
    return output

class AttentionFusion(torch.nn.Module):
    def __init__(self, embed_dim):
        super(AttentionFusion, self).__init__()
        # 可学习的注意力权重
        self.attention_weights = torch.nn.Parameter(torch.randn(2, embed_dim), requires_grad=True)
        
    def forward(self, global_embedding, local_embedding):
        # 计算注意力得分
        global_score = torch.sum(global_embedding * self.attention_weights[0], dim=1, keepdim=True)  # (B, 1)
        local_score = torch.sum(local_embedding * self.attention_weights[1], dim=1, keepdim=True)    # (B, 1)

        # 将注意力得分组合成权重
        scores = torch.cat([global_score, local_score], dim=1)  # (B, 2)
        attention_weights = F.softmax(scores, dim=1)  # 在第 1 维上做 softmax

        # 融合 embedding
        global_weighted = global_embedding * attention_weights[:, 0].unsqueeze(1)  # (B, D)
        local_weighted = local_embedding * attention_weights[:, 1].unsqueeze(1)    # (B, D)

        fused_embedding = global_weighted + local_weighted  # (B, D)
        
        return fused_embedding

class TemporalTransformerConv(nn.Module):
    def __init__(self, embedding_dim):
        super(TemporalTransformerConv, self).__init__()
        self.nout = embedding_dim
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=self.nout, nhead=8), num_layers=6)
        # self.Q = Parameter(torch.ones((args.nout, args.nhid)).to(args.device), requires_grad=True)
        # self.r = Parameter(torch.ones((args.nhid, 1)).to(args.device), requires_grad=True)
        # for p in self.transformer.parameters():
        #     if p.dim() > 1:
        #         glorot(p)
        # glorot(self.Q)
        # glorot(self.r)

    def forward(self, x):
        out = self.transformer(x)
        # last hidden
        history_agg = out[-1,:,:].unsqueeze(0)
        # att pooling
        # att = torch.matmul(torch.tanh(torch.matmul(out, self.Q)), self.r)
        # att = F.softmax(att, dim=0)
        # history_agg = torch.mean(att * out, dim=0).unsqueeze(0)
        return history_agg


class time_encoding(nn.Module):
    # Time Encoding proposed by TGAT
  def __init__(self, dimension):
    super(time_encoding, self).__init__()

    gap=np.array([2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768,65536,131072,262144,524288]) #20个等级
    self.gap = torch.from_numpy(gap)
    self.gap = self.gap
    
  def forward(self, t):
    
    device = t.device
    gap = self.gap.to(device)
    t = t//600
    x = t.shape
    t = t.unsqueeze(1).expand(x[0],gap.shape[0])
    output = torch.sum((t>gap),1).unsqueeze(1)
               
    return output 

class MergeLayer(torch.nn.Module):
  def __init__(self, dim1, dim2, dim3, dim4):
    super().__init__()
    self.fc1 = torch.nn.Linear(dim1 + dim2, dim3)
    self.fc2 = torch.nn.Linear(dim3, dim4)
    self.act = torch.nn.ReLU()

    torch.nn.init.xavier_normal_(self.fc1.weight)
    torch.nn.init.xavier_normal_(self.fc2.weight)

  def forward(self, x1, x2):
    x = torch.cat([x1, x2], dim=1)
    h = self.act(self.fc1(x))
    return self.fc2(h)

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)

        return q, attn
    
class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        return x

class TemporalAttentionLayer(torch.nn.Module):
  """
  Temporal attention layer. Return the temporal embedding of a node given the node itself,
   its neighbors and the edge timestamps.
  """

  def __init__(self, n_node_features, n_neighbors_features, time_dim,
               output_dimension, n_head, n_neighbor,
               dropout=0.1):
    super(TemporalAttentionLayer, self).__init__()

    self.n_head = n_head

    self.feat_dim = n_node_features
    self.time_dim = time_dim

    self.query_dim = n_node_features
    self.key_dim = n_neighbors_features
    
    self.positional_encoding = PositionalEncoding(self.feat_dim, n_neighbor+1)

    self.merger = MergeLayer(self.query_dim, n_node_features, n_node_features, output_dimension)

    self.self_attentions = nn.MultiheadAttention(embed_dim=self.query_dim,
                          kdim=self.key_dim,
                          vdim=self.key_dim,
                          num_heads=8,
                          dropout=dropout)

    self.multi_head_target = nn.MultiheadAttention(embed_dim=self.query_dim,
                                                   kdim=self.key_dim,
                                                   vdim=self.key_dim,
                                                   num_heads=n_head,
                                                   dropout=dropout)

  def forward(self, src_node_features, src_time_features, neighbors_features,
              neighbors_time_features, neighbors_padding_mask):
    
    src_node_features_unrolled = torch.unsqueeze(src_node_features, dim=1)
    
    positions = self.positional_encoding(neighbors_features.size(1)+1).repeat(src_node_features.size(0),1,1) #邻居的位置

    query = src_node_features_unrolled + src_time_features + positions[:,-1,:].unsqueeze(1)
    key = neighbors_features  + neighbors_time_features + positions[:,:-1,:]
    query = query.permute([1, 0, 2])  # [1, batch_size, num_of_features]
    key = key.permute([1, 0, 2])  # [n_neighbors, batch_size, num_of_features]

    # Compute mask of which source nodes have no valid neighbors
    invalid_neighborhood_mask = neighbors_padding_mask.all(dim=1, keepdim=True)
    neighbors_padding_mask[invalid_neighborhood_mask.squeeze(), 0] = False
    
    #enc_output = self.layer_norm(key)

    enc_output, _ = self.self_attentions(key, key, key, key_padding_mask=neighbors_padding_mask)
    key = enc_output
    
    attn_output, attn_output_weights = self.multi_head_target(query=query, key=key, value=key,
                                                              key_padding_mask=neighbors_padding_mask)

    attn_output = attn_output.squeeze()
    attn_output_weights = attn_output_weights.squeeze()

    attn_output = attn_output.masked_fill(invalid_neighborhood_mask, 0)
    attn_output_weights = attn_output_weights.masked_fill(invalid_neighborhood_mask, 0)

    # Skip connection with temporal attention over neighborhood and the features of the node itself
    attn_output = self.merger(attn_output, src_node_features)

    return attn_output, attn_output_weights

class BipartiteGAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=1, dropout=0.6):
        super(BipartiteGAT, self).__init__()
     
        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
       
        self.gat2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout)

        self.dropout = dropout

    def forward(self, x, edge_index):
        # 第一层 GAT 计算用户和项目的表示
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        
        # 第二层 GAT 继续处理
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat2(x, edge_index)
        
        return x
