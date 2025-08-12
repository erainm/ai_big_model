#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project     ：ai_big_model
@File        ：word_embedding_visual_analysis.py
@Create at   ：2025/8/13 15:38
@version     ：V1.0
@Author      ：erainm
@Description : word embedding 可视化分析
TODO：
    1. 对句子分词
    2. 对句子word2id求my_token_list，对句子文本数值化sentence2id
    3. 创建nn.Embedding层，查看每个token的词向量数据
    4. 创建SummaryWriter对象，可视化词向量
        词向量矩阵embd.weight.data和词向量单词列表my_token_list添加到SummaryWriter对象中
        summarywriter.add_embedding(embd.weight.data, my_token_list)
    5. 通过tensorboard观察词向量相似性
    6. 也可以通过程序，从nn.Embedding层中根据idx拿词向量
'''
import jieba
import torch
from tensorflow.keras.preprocessing.text import Tokenizer
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


def wordEmbedding_visual_analysis():
    # 1. 对句子分词
    sentence1 = "从前有座山，山叫无相山，山顶有座破败的观音庙。庙里住着个皱纹能夹死蚊子的老和尚，带着个总爱偷吃供果的小和尚。"
    sentence2 = "小和尚发现师父最近总在子时消失，循着腐臭味来到后山，看见老和尚正用香灰涂抹一具与自己长得一模一样的尸体。供桌上摆着七个褪色的长生牌位，最末一块刻着'释慧明',正是小和尚的法名。"
    sentence3 = "藏经阁的功德簿记载着诡异规律：每隔七年庙里就会多出个'小和尚'，而老和尚的皱纹就会少几道。某夜雷雨交加，小和尚撞见师父在剥一张新鲜人皮，佛龛下的暗格里整整齐齐叠着六套僧袍——每件袖口都绣着'慧明'"
    sentence4 = "当小和尚在镜中看见自己长出第一道皱纹时，山下来了个迷路的少年。老和尚慈祥地摸着新来者的头说：'从今往后，你就叫慧明。'殿内观音像突然流下血泪，香炉里未燃尽的黄纸写着：'第七个，可成佛。'"
    sentences = [sentence1, sentence2, sentence3, sentence4]
    word_list = []
    for s in sentences:
        word_list.append(jieba.lcut(s))
    print("word list --> /n",word_list)
    # 2. 对句子word2id求my_token_list，对句子文本数值化sentence2id
    mytokenizer = Tokenizer()
    mytokenizer.fit_on_texts(texts=word_list)
    print(mytokenizer.word_index)
    # 打印my_token_list
    my_token_list = mytokenizer.index_word.values()
    print("my_token_list --> /n", my_token_list)
    # 打印文本数值化以后的句子
    sentence2id = mytokenizer.texts_to_sequences(texts=word_list)
    print("sentence2id --> /n", sentence2id)
    # 3. 创建nn.Embedding层
    embd = nn.Embedding(num_embeddings=len(my_token_list), embedding_dim=20)
    print("nn.Embedding 层词向量矩阵 --> /n", embd.weight.data, embd.weight.data.shape, type(embd.weight.data))
    # 4. 创建SummaryWriter对象，词向量矩阵embd.weight.data 和词向量单词列表my_token_list
    #   词向量保存到runs目录中，不要出现中文路径（会报错）
    #   log_dir：默认None，保存到当前目录下的run/XXX目录中（自动创建）
    summarywriter = SummaryWriter(log_dir="./runs")
    # mat:词向量表示 张量或numpy数组
    # metadata:词标签
    summarywriter.add_embedding(mat=embd.weight.data, metadata=my_token_list)
    summarywriter.close()

    # 5. 通过tensorboard观察词向量相似性
    # cd 程序的当前目录下执行下面命令
    # 启动tensorboard服务： tensorboard --logdir=runs --host 0.0.0.0
    # 通过浏览器，查看词向量可视化结果 http://127.0.0.1:6006
    print("从nn.Embedding层中根据idx拿词向量")
    # 6. 从nn.Embedding层中根据idx拿词向量
    for idx in range(len(mytokenizer.index_word)):
        tmpvec = embd(torch.tensor(idx))
        print("%4s" % (mytokenizer.index_word[idx + 1]), tmpvec.detach().numpy())

if __name__ == '__main__':
    wordEmbedding_visual_analysis()