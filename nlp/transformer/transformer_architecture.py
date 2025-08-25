# Created by erainm on 2025/8/25 20:17.
# IDE：PyCharm 
# @Project: ai_big_model
# @File：transformer_architecture
# @Description: transformer架构
# TODO:
import torch
import torch.nn as nn
# 数学计算工具包
import math

class Embeddings(nn.Module):
    def __init__(self, vocab, d_model):
        # 参数vocab   词汇表大小
        # 参数d_model 每个词汇的特征尺寸 词嵌入维度
        super(Embeddings, self).__init__()
        self.vocab = vocab
        self.d_model = d_model

        # 定义词嵌入层
        self.embed = nn.Embedding(self.vocab, self.d_model)

    def forward(self, x):
        # 将x传给self.embed并与根号下self.d_model相乘作为结果返回
        # 词嵌入层的权重通常初始化较小（如均匀分布在[-0.1, 0.1]), 导致嵌入后的向量幅度较小。
        # x经过词嵌入后乘以sqrt(d_model)来增大x的值, 与位置编码信息值量纲[-1,1]差不多, 确保两者相加时信息平衡。
        return self.embed(x) * math.sqrt(self.d_model)

def dm_test_Embeddings():
    vocab = 1000  # 词表大小是1000
    d_model = 512  # 词嵌入维度是512维
    # 实例化词嵌入层
    my_embeddings = Embeddings(vocab, d_model)
    x = torch.LongTensor([[100, 2, 421, 508], [491, 998, 1, 221]])
    embed = my_embeddings(x)
    print('embed.shape', embed.shape, '\nembed--->\n', embed)

# 位置编码器类PositionalEncoding 实现思路分析
# 1 init函数  (self, d_model, dropout_p, max_len=5000)
#   super()函数 定义层self.dropout
#   定义位置编码矩阵pe  定义位置列-矩阵position 定义变化矩阵div_term
#   套公式div_term = torch.exp(-torch.arange(0, d_model, 2) / d_model * math.log(10000.0))
#   位置列-矩阵 * 变化矩阵 阿达码积my_matmulres
#   给pe矩阵偶数列奇数列赋值 pe[:, 0::2] pe[:, 1::2]
#   pe矩阵注册到模型缓冲区 pe.unsqueeze(0)三维 self.register_buffer('pe', pe)
# 2 forward(self, x) 返回self.dropout(x)
#   给x数据添加位置特征信息 x = x + self.pe[:,:x.shape[1], :]
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout_p, max_len=5000):
        # 参数d_model 词嵌入维度 eg: 512个特征
        # 参数max_len 单词token个数 eg: 60个单词
        super(PositionalEncoding, self).__init__()

        # 定义dropout层
        self.dropout = nn.Dropout(p=dropout_p)

        # 思路：位置编码矩阵 + 特征矩阵 相当于给特征增加了位置信息
        # 定义位置编码矩阵PE eg pe[60, 512], 位置编码矩阵和特征矩阵形状是一样的
        pe = torch.zeros(max_len, d_model)

        # 定义位置列-矩阵position  数据形状[max_len,1] eg: [0,1,2,3,4...60]^T
        position = torch.arange(0, max_len).unsqueeze(1)
        # print('position--->', position.shape, position)

        # 方式一计算
        _2i = torch.arange(0, d_model, step=2).float()
        pe[:, 0::2] = torch.sin(position / 10000 ** (_2i / d_model))
        pe[:, 1::2] = torch.cos(position / 10000 ** (_2i / d_model))

        # 方式二计算
        # 定义变化矩阵div_term [1,256]
        # torch.arange(start=0, end=512, 2)结果并不包含end。在start和end之间做一个等差数组 [0, 2, 4, 6 ... 510]
        # math.log(10000.0)对常数10000取自然对数
        # torch.exp()指数运算
        # div_term = torch.exp(-torch.arange(0, d_model, 2) / d_model * math.log(10000.0))

        # 位置列-矩阵 @ 变化矩阵 做矩阵运算 [60*1]@ [1*256] ==> 60*256
        # 矩阵相乘也就是行列对应位置相乘再相加，其含义，给每一个列属性（列特征）增加位置编码信息
        # my_matmulres = position * div_term
        # print('my_matmulres--->', my_matmulres.shape, my_matmulres)

        # 给位置编码矩阵奇数列，赋值sin曲线特征
        # pe[:, 0::2] = torch.sin(my_matmulres)
        # 给位置编码矩阵偶数列，赋值cos曲线特征
        # pe[:, 1::2] = torch.cos(my_matmulres)

        # 形状变化 [60,512]-->[1,60,512]
        pe = pe.unsqueeze(0)

        # 把pe位置编码矩阵 注册成模型的持久缓冲区buffer; 模型保存再加载时，可以根模型参数一样，一同被加载
        # 什么是buffer: 对模型效果有帮助的，但是却不是模型结构中超参数或者参数，不参与模型训练
        self.register_buffer('pe', pe)

    def forward(self, x):
        # 注意：输入的x形状2*4*512  pe形状1*60*512  如何进行相加
        # 只需按照x的单词个数 给特征增加位置信息
        x = x + self.pe[:, :x.shape[1], :]
        return self.dropout(x)

def dm_test_PositionalEncoding():
    vocab = 1000  # 词表大小是1000
    d_model = 512  # 词嵌入维度是512维

    # 1 实例化词嵌入层
    my_embeddings = Embeddings(vocab, d_model)

    # 2 让数据经过词嵌入层 [2,4] --->[2,4,512]
    x = torch.LongTensor([[100, 2, 421, 508], [491, 998, 1, 221]])
    embed = my_embeddings(x)
    # print('embed--->', embed.shape)

    # 3 创建pe位置矩阵 生成位置特征数据[1,60,512]
    my_pe = PositionalEncoding(d_model=d_model, dropout_p=0.1, max_len=60)

    # 4 给词嵌入数据embed 添加位置特征 [2,4,512] ---> [2,4,512]
    pe_result = my_pe(embed)
    print('pe_result.shape--->', pe_result.shape)
    print('pe_result--->', pe_result)

import matplotlib.pyplot as plt
import numpy as np

# 绘制PE位置特征sin-cos曲线
def dm_draw_PE_feature():
    # 1 创建pe位置矩阵[1,5000,20]，每一列数值信息：奇数列sin曲线 偶数列cos曲线
    my_pe = PositionalEncoding(d_model=20, dropout_p=0)
    print('my_positionalencoding.shape--->', my_pe.pe.shape)

    # 2 创建数据x[1,100,20], 给数据x添加位置特征  [1,100,20] ---> [1,100,20]
    y = my_pe(torch.zeros(1, 100, 20))
    print('y--->', y.shape)

    # 3 画图 绘制pe位置矩阵的第4-7列特征曲线
    plt.figure(figsize=(20, 10))
    # 第0个句子的，所有单词的，绘制4到8维度的特征 看看sin-cos曲线变化
    plt.plot(np.arange(100), y[0, :, 4:8].numpy())
    plt.legend(["dim %d" % p for p in [4, 5, 6, 7]])
    plt.show()

    # print('直接查看pe数据形状--->', my_pe.pe.shape) # [1,5000,20]
    # 直接绘制pe数据也是ok
    # plt.figure(figsize=(20, 20))
    # # 第0个句子的，所有单词的，绘制4到8维度的特征 看看sin-cos曲线变化
    # plt.plot(np.arange(100), my_pe.pe[0,0:100, 4:8])
    # plt.legend(["dim %d" %p for p in [4,5,6,7]])
    # plt.show()


if __name__ == '__main__':
    # dm_test_Embeddings()
    # dm_test_PositionalEncoding()
    dm_draw_PE_feature()