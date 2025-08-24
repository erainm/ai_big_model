# Created by erainm on 2025/8/24 11:50.
# IDE：PyCharm 
# @Project: ai_big_model
# @File：add_attention_mechanism
# @Description: 加性注意力机制
# TODO: 自注意力机制

import torch
import torch.nn as nn
import torch.nn.functional as func


class AdditiveAttention(nn.Module):
    """
    加性注意力机制
    使用加性计算方法，对Q, K, V进行注意力计算
    """
    def __init__(self, hidden_dim, attention_dim=None):
        """
        初始化加性注意力层
        参数:
        hidden_dim: Q, K, V的隐藏维度
        attention_dim: 注意力空间的维度，默认为hidden_dim
        """
        super(AdditiveAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.attention_dim = attention_dim or hidden_dim

        # 第一步：将Q和K映射到注意力空间
        self.W_q = nn.Linear(hidden_dim, self.attention_dim, bias=False)
        self.W_k = nn.Linear(hidden_dim, self.attention_dim, bias=False)

        # 第二步：加性注意力计算参数
        self.v = nn.Linear(self.attention_dim, 1, bias=False)

        # 第三步：输出线性变换层
        self.W_o = nn.Linear(hidden_dim, hidden_dim)

        # 用于存储注意力权重
        self.attention_weights = None

    def forward(self, Q, K, V):
        """
        前向传播

        参数:
        Q: 查询张量 (batch_size, seq_len_q, hidden_dim)
        K: 键张量 (batch_size, seq_len_k, hidden_dim)
        V: 值张量 (batch_size, seq_len_v, hidden_dim)

        返回:
        output: 注意力输出 (batch_size, seq_len_q, hidden_dim)
        attention_weights: 注意力权重 (batch_size, seq_len_q, seq_len_k)
        """
        batch_size, seq_len_q, _ = Q.size()
        seq_len_k = K.size(1)

        # 第一步：将Q和K映射到注意力空间
        Q_transformed = self.W_q(Q)  # (batch_size, seq_len_q, attention_dim)
        K_transformed = self.W_k(K)  # (batch_size, seq_len_k, attention_dim)

        # 扩展维度以便广播相加
        Q_expanded = Q_transformed.unsqueeze(2)  # (batch_size, seq_len_q, 1, attention_dim)
        K_expanded = K_transformed.unsqueeze(1)  # (batch_size, 1, seq_len_k, attention_dim)

        # 加性注意力计算: tanh(W_q * Q + W_k * K)
        additive = torch.tanh(Q_expanded + K_expanded)  # (batch_size, seq_len_q, seq_len_k, attention_dim)

        # 计算注意力分数
        attention_scores = self.v(additive).squeeze(-1)  # (batch_size, seq_len_q, seq_len_k)


        # 计算注意力权重
        attention_weights = func.softmax(attention_scores, dim=-1)
        self.attention_weights = attention_weights  # 存储用于可视化

        # 应用注意力权重到V
        # attention_weights: (batch_size, seq_len_q, seq_len_k)
        # V: (batch_size, seq_len_k, hidden_dim)
        output = torch.bmm(attention_weights, V)  # (batch_size, seq_len_q, hidden_dim)

        # 第三步：线性变换得到最终输出
        output = self.W_o(output)

        return output, attention_weights


# 自注意力版本（Q, K, V相同）
class SelfAdditiveAttention(AdditiveAttention):
    """
    自注意力机制的加性注意力版本
    Q = K = V
    """

    def forward(self, x):
        """
        参数:
        x: 输入张量 (batch_size, seq_len, hidden_dim)
        mask: 掩码张量 (batch_size, seq_len, seq_len)
        """
        return super().forward(x, x, x)


# 示例用法
if __name__ == "__main__":
    # 参数设置
    batch_size = 2
    seq_len = 5
    hidden_dim = 64
    attention_dim = 128

    # 创建注意力机制实例
    attention = AdditiveAttention(hidden_dim, attention_dim)

    # 创建随机输入数据
    Q = torch.randn(batch_size, seq_len, hidden_dim)
    K = torch.randn(batch_size, seq_len + 2, hidden_dim)  # 键值序列可以不同长度
    V = torch.randn(batch_size, seq_len + 2, hidden_dim)

    # 前向传播
    output, attn_weights = attention(Q, K, V)

    print(f"Q形状: {Q.shape}")
    print(f"K形状: {K.shape}")
    print(f"V形状: {V.shape}")
    print(f"输出形状: {output.shape}")
    print(f"注意力权重形状: {attn_weights.shape}")

    # 自注意力
    print("\n自注意力示例:")
    self_attention = SelfAdditiveAttention(hidden_dim, attention_dim)
    x = torch.randn(batch_size, seq_len, hidden_dim)
    output_self, attn_weights_self = self_attention(x)

    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output_self.shape}")
    print(f"注意力权重形状: {attn_weights_self.shape}")