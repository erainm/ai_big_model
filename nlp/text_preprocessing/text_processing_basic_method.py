#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project     ：ai_big_model
@File        ：text_processing_basic_method.py
@Create at   ：2025/8/12 14:01
@version     ：V1.0
@Author      ：erainm
@Description : 文本处理的基本方法

TODO：
    分词：将连续的字序列按照一定的规范重新组合成词序列的过程
    作用：
        预处理：分词是文本处理的第一步，能够将文本分解为有意义的单元，为后续的分析提供基础
        理解结构:分词有助于理解句子的基本构成和含义，尤其在做文本分类、情感分析等任务时，分词时不可缺少的一步
    常用分词工具：Jieba、THULAC、HanLP等
    Jieba分词工具：
        一个开源的组件，支持精确模式、全模式和搜索引擎模式三种分词模式
        主要特点：
            · 支持多种分词模式，满足不同的场景需求
            · 支持自定义词典：用户可添加自定义词语，提高分词准确率
            · 支持词性标注：可以为每个词语标注词性，例如：名词、动词
            · 支持关键词提取：可以提取文本中的关键词
            · 支持并行分析：可以利用多核处理器加速分词
            · 简单易用：API简单、易于上手
            · 开源免费
        安装： pip install jieba -i https://pypi.mirrors.ustc.edu.cn/simple/
'''
import jieba

# 精确模式分词，默认模式，适合文本分类、信息检索、文本摘要、文本聚类等场景
content = "君不见黄河之水天上来，奔流到海不复回。君不见高堂明镜悲白发，朝如青丝暮成雪。人生得意须尽欢，莫使金樽空对月。天生我材必有用，千金散尽还复来。烹羊宰牛且为乐，会须一饮三百杯。岑夫子，丹丘生，将进酒，杯莫停。与君歌一曲，请君为我倾耳听。钟鼓馔玉不足贵，但愿长醉不愿醒。古来圣贤皆寂寞，惟有饮者留其名。陈王昔时宴平乐，斗酒十千恣欢谑。主人何为言少钱，径须沽取对君酌。五花马、千金裘，呼儿将出换美酒，与尔同销万古愁。"
print("-----------------------精确模式分词--------------------------")
print(jieba.cut(sentence=content, cut_all=False)) # cut_all=False 默认精确模式
print(jieba.lcut(sentence =  content, cut_all=False))

# 全模式分词：将句子中的所有可以成词的词语都扫描出来，速度非常快，但不能消除歧义
print("-----------------------全模式分词--------------------------")
print(jieba.lcut(sentence=content, cut_all=True))

# 搜索引擎模式分词：在精确模式的基础上，对长词再次切分，提高召回率，适合用于搜索引擎分词
print("-----------------------搜索引擎模式分词--------------------------")
print(jieba.cut_for_search(sentence=content))
print(jieba.lcut_for_search(sentence=content))

# 中文繁体分词，针对香港、台湾等地区
print("-----------------------中文繁体分词--------------------------")
content1 = "煩惱即是菩提，我暫且不提"
print(jieba.lcut(sentence=content1, cut_all=False, HMM=True))


# 自定义词典
print("-----------------------自定义词典--------------------------")
jieba.load_userdict("dict.txt")
print(jieba.lcut(sentence=content, cut_all=False))

# 命名实体识别
"""
    命名实体识别（NER）：是自然语言处理中的一个任务，旨在从文本中识别出特定类别的实体（如：人名、地名、日期、时间等）。
    NER是信息抽取的一部分，帮助计算机识别出与任务相关信息的实体信息。
    作用：
        · 信息抽取：NER帮助从海量文本中自动抽取结构化的实体信息，为数据分析、问答系统等提供有价值的内容
        · 问答系统：在只能问答系统中，NER能够宝珠系统准确理解用户的提问，并提取相关的实体信息以便生成更准确的回答。
        · 文本理解：NER对于文本理解至关重要，帮助系统识别出文本中的关键信息，例如任务、时间、地点等，进而为语义分析和事件抽取提供支持
    处理工具：
        SpaCy、NLTK、Stanford NER、BERT(通过微调)、LTP、HanLP等都可以用于命名实体识别任务。
    安装：pip install hanlp_restful
"""
from hanlp_restful import HanLPClient
# 创建客户端
client = HanLPClient(url="https://www.hanlp.com/api", auth=None, language="zh")
text = "杜甫是唐代伟大的现实主义诗人，被誉为“诗圣”，其诗作深刻反映社会现实与人民疾苦，代表作有三吏、三别等。"
# 进行命名实体识别
# 只选择一个任务，包含分词和命名实体识别
result = client.parse(text, tasks=["ner/msra"])
print(result)

# 词性标注
"""
    词性标注（Part-Of-Speech tagging,简称POS）：为文本中的每个词分配一个语法类别（词性），例如：名词、动词、形容词等。
    词性标注能够帮助模型理解词汇在句子中的语法功能，并进一步的句法分析和语义分析提供支持。
    类型：
        · 名词(n)：表示人、事物、地方等
        · 动词(v)：表述动作、存在等
        · 形容词(a)：描述事物性质或状态
        · 副词(d)：修饰动词、形容词或其它副词
        · 代词(r)：代替名词的词
    作用：
        · 理解句子结构：通过词性标注，知道每个词在句子中的角色，帮助理解句子的语法结构
        · 支持其它NLP任务：许多高级任务，如命名实体识别（NER）、句法分析、情感分析等，通常依赖于词性标注的结果。
        · 歧义消除：词性标注有助于解决同一单词在不同上下文中可能具有不同词性的情况。例如，单词lead，可能是动词(引导)也可能是名词(铅)，通过词性标注可以解决这种歧义
    处理工具：Jieba、NLTK、SpaCy、Stanford POS Tagger等是常用的词性标注工具
"""
print("-----------------------词性标注--------------------------")
import jieba.posseg as pseg

text2 = "我爱北京天安门"
# 步骤1：分词并进行词性标注
# 结果返回一个装有pair元组的列表, 每个pair元组中分别是词汇及其对应的词性, 具体词性含义请参照[附录: jieba词性对照表]()
words = pseg.lcut(text2)
print("words --> ", words)
# 提取命名实体（如：人名、地名等）
name_entities = []
for word, flag in words:
    if flag in ["n", "nr", "ns", "nt"]:
        name_entities.append(word)
print("name_entities --> ", name_entities)
