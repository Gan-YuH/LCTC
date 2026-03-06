import torch
import torch.nn as nn
import numpy as np
import math
from torch.nn.parameter import Parameter


class Conv1D(nn.Module):
    def __init__(self, out_dim, rf, in_dim):
        super(Conv1D, self).__init__()
        self.rf = rf
        self.out_dim = out_dim
        if rf == 1:
            w = torch.empty(in_dim, out_dim)
            nn.init.normal_(w, std=0.02)
            self.w = Parameter(w)
            self.b = Parameter(torch.zeros(out_dim))
        else:
            raise NotImplementedError

    def forward(self, x):
        if self.rf == 1:
            # print('bbbb:',x.shape)
            size_out = x.size()[:-1] + (self.out_dim,)
            # x = torch.addmm(self.b, x.view(-1, x.size(-1)), self.w)
            x = torch.addmm(self.b, x.reshape(-1, x.size(-1)), self.w)

            x = x.view(*size_out)
        else:
            raise NotImplementedError
        return x

class LogSparseAttention(nn.Module):
    """
    Args:
        n_time_series: Number of time series present in input
        n_head: Number of heads in the MultiHeadAttention mechanism
        seq_num: The number of targets to forecast
        sub_len: sub_len of the sparse attention
        num_layer: The number of transformer blocks in the model.
        n_embd: The dimention of Position embedding and time series ID embedding
        forecast_history: The number of historical steps fed into the time series model
        dropout: The dropout for the embedding of the model.
        additional_params: Additional parameters used to initalize the attention model. Can inc
    """

    def __init__(self, n_head, n_embd, win_len, scale: bool, q_len: int, sub_len, sparse=True, attn_pdrop=0.1,
                 resid_pdrop=0.1):
        super(LogSparseAttention, self).__init__()

        if sparse:
            # print('Activate log sparse!')
            mask = self.log_mask(win_len, sub_len)
        else:
            mask = torch.tril(torch.ones(win_len, win_len)).view(1, 1, win_len, win_len)

        self.register_buffer('mask_tri', mask)
        self.n_head = n_head
        self.split_size = n_embd * self.n_head
        self.scale = scale
        self.q_len = q_len
        self.query_key = nn.Conv1d(n_embd, n_embd * n_head * 2, self.q_len)
        self.value = Conv1D(n_embd * n_head, 1, n_embd)
        self.c_proj = Conv1D(n_embd, 1, n_embd * self.n_head)
        self.attn_dropout = nn.Dropout(attn_pdrop)
        self.resid_dropout = nn.Dropout(resid_pdrop)

    def log_mask(self, win_len, sub_len):
        mask = torch.zeros((win_len, win_len), dtype=torch.float)
        for i in range(win_len):
            mask[i] = self.row_mask(i, sub_len, win_len)
        return mask.view(1, 1, mask.size(0), mask.size(1))

    def row_mask(self, index, sub_len, win_len):
        """
        Remark:
        1 . Currently, dense matrices with sparse multiplication are not supported by Pytorch. Efficient implementation
            should deal with CUDA kernel, which we haven't implemented yet.

        2 . Our default setting here use Local attention and Restart attention.

        3 . For index-th row, if its past is smaller than the number of cells the last
            cell can attend, we can allow current cell to attend all past cells to fully
            utilize parallel computing in dense matrices with sparse multiplication."""
        log_l = math.ceil(np.log2(sub_len))
        mask = torch.zeros((win_len), dtype=torch.float)
        if ((win_len // sub_len) * 2 * (log_l) > index):
            mask[:(index + 1)] = 1
        else:
            while (index >= 0):
                if ((index - log_l + 1) < 0):
                    mask[:index] = 1
                    break
                mask[index - log_l + 1:(index + 1)] = 1  # Local attention
                for i in range(0, log_l):
                    new_index = index - log_l + 1 - 2 ** i
                    if ((index - new_index) <= sub_len and new_index >= 0):
                        mask[new_index] = 1
                index -= sub_len
        return mask

    def attn(self, query: torch.Tensor, key, value: torch.Tensor):
        activation = nn.Softmax(dim=-1)
        pre_att = torch.matmul(query, key)
        if self.scale:
            pre_att = pre_att / math.sqrt(value.size(-1))
        mask = self.mask_tri[:, :, :pre_att.size(-2), :pre_att.size(-1)]
        pre_att = pre_att * mask + -1e9 * (1 - mask)
        pre_att = activation(pre_att)
        pre_att = self.attn_dropout(pre_att)
        attn = torch.matmul(pre_att, value)

        return attn

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)
        if k:
            return x.permute(0, 2, 3, 1)
        else:
            return x.permute(0, 2, 1, 3)

    def forward(self, x):
        # print('xxxxxxxxxxxxxxx:',x.shape)
        value = self.value(x)
        qk_x = nn.functional.pad(x.permute(0, 2, 1), pad=(self.q_len - 1, 0))
        query_key = self.query_key(qk_x).permute(0, 2, 1)
        query, key = query_key.split(self.split_size, dim=2)
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)
        attn = self.attn(query, key, value)
        attn = self.merge_heads(attn)
        attn = self.c_proj(attn)
        attn = self.resid_dropout(attn)
        return attn

# 位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

# 4. 时序卷积网络
class TCN(nn.Module):
    def __init__(self, d_model, kernel_size=3, dilation=1, dropout=0.1):
        super(TCN, self).__init__()
        self.conv = nn.Conv1d(
            d_model, d_model,
            kernel_size,
            padding=dilation * (kernel_size - 1) // 2,
            dilation=dilation
        )
        self.norm = nn.LayerNorm(d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        x_conv = self.conv(x.transpose(1, 2)).transpose(1, 2)
        x_norm = self.norm(x + self.dropout(self.relu(x_conv)))
        return x_norm

# 自定义 Encoder 层（突出残差连接）
class CustomEncoderLayer(nn.Module):
    def __init__(self, win_len,d_model, nhead, dim_feedforward=256, dropout=0.1):
        super().__init__()
        # self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.self_attn = LogSparseAttention(nhead, d_model,
                                            win_len=win_len,
                                            scale=True,
                                            q_len=5,
                                            sub_len=12,)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.TCN = TCN(d_model)

    def forward(self, src):
        # Self-Attention + 残差连接
        # print(f"Before Self-Attention: {src.shape}")
        # attn_output, _ = self.self_attn(src, src, src)  # [seq_len, batch_size, d_model]

        attn_output = self.self_attn(src.transpose(0, 1))  # [batch_size, seq_len, d_model]
        # print(f"After Self-Attention: {attn_output.shape}")
        attn_output = attn_output.transpose(0, 1)  # [batch_size, seq_len, d_model]

        src = self.norm1(src + self.dropout1(attn_output))  # 残差连接
        # print(f"After Self-Attention residual: {src.shape}")
        src = self.TCN(src)

        # FFN + 残差连接
        ffn_output = self.linear2(self.dropout(torch.relu(self.linear1(src))))
        src = self.norm2(src + self.dropout2(ffn_output))  # 残差连接
        # print(f"After FFN residual: {src.shape}")
        return src


# Transformer 模型
class TransformerTimeSeries(nn.Module):
    def __init__(self, input_size=1, input_seq=7, hidden_size=64, num_layers=2, num_heads=4, out_seq=28):
        '''
        :param input_size:输入特征维度
        :param hidden_size: 编码维度
        :param num_layers:
        :param num_heads: 多头注意力头数
        :param out_size: 输出序列维度
        '''
        super().__init__()
        self.input_embedding = nn.Linear(input_size, hidden_size)
        self.pos_encoder = PositionalEncoding(hidden_size)
        self.encoder_layers = nn.ModuleList([CustomEncoderLayer(input_seq,hidden_size, num_heads) for _ in range(num_layers)])
        self.fc_out = nn.Linear(hidden_size, input_size)
        self.input_size = input_size
        self.input_seq = input_seq
        self.out_seq = out_seq

    def forward(self, src):
        # print('input:',src.shape)   # [batch_size, seq_len, feature-dim]
        src = self.input_embedding(src)  # [batch_size, 2016, hidden_size]
        # print('input_embedding:',src.shape)
        src = self.pos_encoder(src.transpose(0, 1)).transpose(0, 1)
        # print('pos_encoder:', src.shape)

        for layer in self.encoder_layers:
            src = layer(src.transpose(0, 1)).transpose(0, 1)  # [batch_size, seq_len, hidden_size]

        if self.out_seq > self.input_seq:
            print('self.out_seq:',self.out_seq)
        else:
            src = src[:, -self.out_seq:, :]
        return self.fc_out(src)  # [batch_size, 288, 7]


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    feture_dim = 1
    input_seq = 256
    out_seq = 1
    x = torch.rand(1, input_seq, feture_dim).to(device)
    model = TransformerTimeSeries(input_size=feture_dim, input_seq=input_seq, out_seq = out_seq).to(device)

    y = model(x)
    print('out:',y.shape)





    print('our :')
    from thop import profile
    flops, params = profile(model, inputs=(x,))
    print("FLOPs:   ", flops)
    print("params:   ", params)

    print('推理时间（Latency）')
    import time
    model.eval()
    with torch.no_grad():
        start = time.time()
        for _ in range(100):
            y = model(x,)
        end = time.time()
    print("GPU Average inference time:", (end - start)/100)

    print('吞吐量（Throughput）')
    start = time.time()
    for _ in range(100):
        y = model(x,)
    end = time.time()

    throughput = 100 * 1 / (end - start)
    print("Throughput (samples/sec):", throughput)

    print('GPU 显存占用')
    print("Max memory allocated (MB):",
        torch.cuda.max_memory_allocated()/1024/1024)

