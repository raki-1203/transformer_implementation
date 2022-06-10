import torch
import torch.nn as nn

from model.utils import initialize_weight


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, dropout_rate, head_size=8):
        super(MultiHeadAttention, self).__init__()

        self.head_size = head_size

        self.att_size = att_size = hidden_size // head_size
        self.scale = att_size ** -0.5

        self.linear_q = nn.Linear(hidden_size, head_size * att_size, bias=False)
        self.linear_k = nn.Linear(hidden_size, head_size * att_size, bias=False)
        self.linear_v = nn.Linear(hidden_size, head_size * att_size, bias=False)
        initialize_weight(self.linear_q)
        initialize_weight(self.linear_k)
        initialize_weight(self.linear_v)

        self.att_dropout = nn.Dropout(dropout_rate)

        self.output_layer = nn.Linear(head_size * att_size, hidden_size, bias=False)
        initialize_weight(self.output_layer)

    def forward(self, q, k, v, mask, cache=None):
        orig_q_size = q.size()

        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)

        # head_i = Attention(Q(W^Q)_i, K(W^K)_i, V(W^V)_i)
        q = self.linear(q).view(batch_size, -1, self.head_size, d_k)  # [batch_size, q_len, head, d_k]
        if cache is not None and 'encdec_k' in cache:
            k, v = cache['encdec_k'], cache['encdec_v']
        else:
            k = self.linear_k(k).view(batch_size, -1, self.head_size, d_k)  # [batch_size, k_len, head, d_k]
            v = self.linear_v(v).view(batch_size, -1, self.head_size, d_v)  # [batch_size, v_len, head, d_k]

            if cache is not None:
                cache['encdec_k'], cache['encdec_v'] = k, v

        q = q.transpose(1, 2)  # [batch_size, head, q_len, d_k]
        v = v.transpose(1, 2)  # [batch_size, head, v_len, d_v]
        k = k.transpose(1, 2).transpose(2, 3)  # [batch_size, head, d_k, k_len]

        # Scaled Dot-Product Attention
        # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
        q.mul_(self.scale)
        x = torch.matmul(q, k)  # [batch_size, head, q_len, k_len]
        x.masked_fill_(mask.unsqueeze(1), -1e9)
        x = torch.softmax(x, dim=3)
        x = self.att_dropout(x)
        x = x.matmul(v)  # [batch_size, head, q_len, d_v]

        x = x.transpose(1, 2).contiguous()  # [batch_size, q_len, head, d_v]
        x = x.view(batch_size, -1, self.head_size * d_v)  # [batch_size, q_len, hidden_size)

        x = self.output_layer(x)

        assert x.size() == orig_q_size
        return x
