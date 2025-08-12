#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project     ：ai_big_model
@File        ：text_tensor.py
@Create at   ：2025/8/12 15:40
@version     ：V1.0
@Author      ：erainm
@Description : 文本张量表示
'''
import torch

from nlp.text_preprocessing.text_processing_basic_method import result

"""
    文本张量表示：将一段文本使用张量表述，其中一般将词汇表示成向量，称作词向量，再由各个词向量按顺序组成矩阵形成文本表述。
    作用：将文本表示成张量(矩阵)形式，能够使语言文本可以作为计算机处理程序的输入，进行接下来一系列的解析工作。
        · 连接文本与计算机
            - 机器可读性：计算机无法直接理解人类语言，文本张量表示是将文本转换为数值形式（通常是多维数组），使其能够被计算及处理和理解
            - 模型输入：大多数机器学习和深度学习模型（包括NLP）都需要数值输入，文本张量表示是文本数据进入模型的桥梁
        · 表达语义信息
            - 捕捉词语关系：好的文本张量表示方法，例如词嵌入，可以将词语映射到高维空间中，使得语义相似的词语在向量空间中也彼此接近。
                例如：king和queen的向量在空间中会比king和apple的更接近
            - 保留上下文信息：对于句子和文档的表示方法，例如句嵌入和文档嵌入，能够保留文本上下文信息，例如词语之间的顺序和依赖关系。
            - 理解文本含义：通过将文本映射到向量空间，模型可以学习到文本的深层语义含义，而不仅仅是表面的字面意思。
        · 提升模型性能：
            - 特征提取：文本张量表示可以看作是对文本进行特征提取的过程，将文本转为计算机可以理解的特征。
            - 降维：一些文本张量表示方法，例如词嵌入，可以将文本的维度降低，减少模型的计算量，并避免维度灾难。
            - 减少噪声：一些文本张量表示方法，例如：TF-IDF，可以对文本中的噪声进行过滤，突出重要信息。
    文本张量的表示方法：
        · one-hot编码
        · Word2vec
        · Word Embedding
        · ……
"""

"""
    one-hot词向量表示
        one-hot编码是一种将离散的分类变量转化为二进制向量的方法。在自然语言处理中，one-hot编码常用于表示单词。
        每个单词都被表示为一个稀疏向量，该向量的长度等于词汇表的大小，其中只有一个位置为1，其它位置为0.
        例如：一个单词词汇表["cat", "dog", "fish"]
            "cat" --> [1, 0, 0]
            "dog" --> [0, 1, 0]
            "fish"--> [0, 0, 1]
        特点：
            · 稀疏性：one-hot编码通常会产生非常稀疏的向量，尤其是词汇表很大时。大部分元素为零，只有一个位置是1.
            · 维度较高：词汇表的大小决定了one-hot向量的维度，如果词汇表包含10000个单词，那么每个单词的表示将是一个长度为10000的向量
            · 信息丢失：one-hot编码无法表达词与词之间的语义关系。例如：cat和dog的表示完全不同，尽管语义上很接近
        优缺点：
            优点：实现简单，容易理解
            缺点：高维稀疏向量，计算效率低，无法捕捉词之间的语义相似性。而且在大语料集下，每个向量的长度过大，占据大量内存
"""

# 导入keras中的词汇映射器Tokenizer
from tensorflow.keras.preprocessing.text import Tokenizer
# pip install tensorflow==2.12.1 -i https://mirrors.aliyun.com/pypi/simple/
# 导入用于对象保存与加载的joblib
import joblib

def text_one_hot():
    # 1. create tokenizer
    vocabs = ["周杰伦", "陈奕迅", "林俊杰", "薛之谦", "张学友", "刘德华"]
    # 2. init tokenizer
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(vocabs)
    print("word index mapping",tokenizer.word_index)
    print("index word mapping",tokenizer.index_word)
    # 3. use 'texts_to_matrix' to convert text to matrix(one-hot code)
    one_hot_matrix = tokenizer.texts_to_matrix(vocabs, mode="binary")[:, 1:]
    print("one_hot_matrix -->", one_hot_matrix)

    print("------one-hot 编码--------------")
    for word, vector in zip(vocabs, one_hot_matrix):
        # numpy vector float convert to integer list
        result = vector.astype(int).tolist()
        print(f"{word} one-hot code is: {result}")
    # 4. use joblib save Tokenizer object,can repeat use
    tokenizer_path = "./onehot_tokenizer.joblib"
    joblib.dump(tokenizer,tokenizer_path)

    # 5. use joblib load Tokenizer object,and get "周杰伦" one-hot code
    print("--- One-Hot code ---")
    tokenizer = joblib.load(tokenizer_path)
    result_array = tokenizer.texts_to_matrix(['周杰伦'], mode='binary')[0, 1:]
    result_list = result_array.astype(int).tolist()
    print(f"周杰伦 one-hot code is: {result_list}")

if __name__ == '__main__':
    text_one_hot()


"""
    Word2Vec模型
    概念:
        · Word2Vec是一种流行的将词汇表示成向量的无监督训练方法，该过程将构建神经网络模型，
            将网络参数作为词汇的向量表示，通过上下文信息来学习词语的分布表示（即词向量）。它包括CBOW和skipgram两种训练模式。
        · Word2Vec实际上利用了文本本身的信息来构建“伪标签”。模型不是被人为的告知某个词语的正确词向量，
            而是通过上下文词语来预测中心词（CBOW）或者通过中心词来预测上下文词语（Skip-gram）。
        · Word2Vec的目标是将词转换为一个固定长度的向量，这些向量能够捕捉词与词之间的语义关系。
    特点：
        · 密集表示：Word2Vec通过训练得到的词向量通常是密集的，即大部分值不为0，每个向量的维度较小（通常几十到几百维）。
        · 捕捉语义关系：Word2Vec可以通过词向量捕捉到词之间的语义相似性，例如通过向量运算，可以发现king - man + woman ≈ queen
    优缺点：
        · 优点：能够生成稠密的词向量，捕捉词与词之间的语义关系，计算效率高。
        · 缺点：需要大量的语料来训练，且可能不适用于某些特定任务（如：词语的多义性）
"""

"""
    CBOW(Continuous bag of words)模式
        · 概念： 给定一段用于训练的上下文词汇，预测目标词汇
        · CBOW模式下的Word2Vec过程
    
    Skip-gram模式
        · 概念：给定一个目标词，预测其上下文词汇。
"""

"""
    词嵌入Word Embedding
    Word Embedding与Word2Vec的关系
        · Word2Vec是一种Word Embedding方法，专门用于生成词的稠密向量表示。
            Word2Vec通过神经网络训练，利用上下文信息将每个词表示为一个低维稠密向量。
        · Word Embedding是一个更广泛的概念，指任何将词汇映射到低维空间的表示方法，不仅限于Word2Vec。GloVe和FastText等也属于词嵌入
    Word Embedding概念：
        · 一种通过一定的方式将单词映射到指定维度的空间技术，目的是将单词的语义信息编码进低维向量空间。
        · 广义的word embedding包括所有密集词汇向量的表示方法，如word2vec，即可认为是word embedding的一种。
        · 侠义的word embedding是指在神经网络中加入的embedding层，对整个网络进行训练的同时产生的embedding矩阵(embedding层的参数)，
            这个embedding矩阵就是训练过程中所有输入词汇的向量表示组成的矩阵。
    特点：
        · 低维稠密向量：词嵌入将每个词映射为低维稠密向量，通常维度50、100、200等
        · 捕捉语义和句法信息：词嵌入能够捕捉词语之间的关系，例如语法上的相似性（如复数形式）和语义上的相似性（man和woman）
        · 迁移学习：词嵌入能够在不同任务之间共享词向量，提高模型的泛化能力。
    优缺点：
        · 优点：能够有效捕捉词语的语义和句法信息，且训练出来的词向量可以在多个任务中使用。
        · 缺点：对于一些低频词和未见过的词处理可能较差。
"""

import torch.nn as nn
import jieba

# initialization text
text = "我是中国人，我爱中国"
# 1. instance embedding layer object
# num_embeddings: 词表大小，词表中有多少个词，当前就设置多少
# embedding_dim：词向量的维度数，自定义
embedding = nn.Embedding(num_embeddings = 8,embedding_dim = 5)
print("embedding --> ",embedding)
# 2. 将文本进行jieba分词
word_list = jieba.lcut(text)
print("word_list --> ",word_list)
# 3. 获取词对应的词下标
word_index = [word_list.index(word) for word in word_list]
print("word_index --> ",word_index)
# 4. 将词下标转化为张量（embedding需要的是张量类型）
word_index_tensor = torch.tensor(word_index)
print("word_index_tensor --> ", word_index_tensor)
# 5. 使用word embedding将词下标(张量类型) 向量化
result = embedding(word_index_tensor)
print("result --> ", result)