# Created by erainm on 2025/8/23 14:47.
# IDE：PyCharm 
# @Project: ai_big_model
# @File：english2french
# @Description: 英语翻译法语
# TODO:

# 正则表达式包
import re

# 构建网络结构和函数的torch包
import torch
import torch.nn as nn
from torch.cuda import device
from torch.utils.data import DataLoader,Dataset

# torch中的优化方法工具包
import torch.optim as optim
import time

# 随机生成数据的包
import random
import matplotlib.pyplot as plt
import numpy as np

# 设备选择(选择在cuda或者cpu上运行代码)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 设置起始标志 SOS --> start of sequence
SOS_token = 0
# 设置结束标志 EOS --> end of sequence
EOS_token = 1
# 设置句子最大长度不超过10(包括标点),用于设置每个句子样本中间语义张量c长度都为10
MAX_LENGTH = 10
# 数据文件路径
data_path = "./data/eng-fra-v2.txt"
# 模型训练参数
mylr = 1e-4
epochs = 2
print_interval_num = 1000
plot_interval_num = 100


# 准备一: 文本清洗工具函数
def normalizeString(s:str):
    '''字符串规范化函数, 参数s表示传入字符串'''
    s = s.lower().strip()
    # 在.!?前加一个空格, 即用 “空格 + 原标点” 替换原标点。
    # \1 代表 捕获的标点符号，即 ., !, ? 之一。
    s = re.sub(r"([.!?])", r" \1", s)
    # 用一个空格替代原标点,即:标点符号完全去掉,只留空格
    s = re.sub(r"([.!?])", r" ", s)
    # 使用正则表达式将字符串中 不是 至少1个小写字母和正常标点的都替换成空格
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    # 再次清理多余的空格
    s = re.sub(r"\s+", " ", s).strip()
    return s
# -------------------------------------------------步骤一:获取数据,并初步处理-------------------------------------------------
# 步骤一:获取数据,并处理数据
def getdata():
    # 1. 按行读取文件
    with open(data_path, "r", encoding="utf-8") as f:
        lines = f.read().strip().split("\n")
    print("lines length --> ", len(lines))

    # 2. 按行清洗文本,构建语言对my_pairs
    # 格式: [['英文句子','法语句子'], ['英文句子', '法语句子']]
    tmp_pairs, my_pairs = [],[]
    for l in lines:
        for s in l.split("\t"):
            tmp_pairs.append(normalizeString(s))
        my_pairs.append(tmp_pairs)
        # 清空tmp_pairs, 以便于存储下一个英语和法语的句子对
        tmp_pairs = []

    # 3. 遍历语言对,构建英语单词字典和法语单词字典
    # 字典位置0和1 存放的是开始标志和结束标志
    english_word2index = {'SOS':0, 'EOS':1}
    french_word2index = {'SOS':0, 'EOS':1}
    # 单词从下标2开始
    english_word_n = 2
    french_word_n = 2
    # 遍历语言对 获取英语单词字典 法语单词字典
    for pair in my_pairs:
        for word in pair[0].split(" "):
            if word and word not in english_word2index:
                english_word2index[word] = english_word_n
                # 更新下一个单词的下标值
                english_word_n += 1
    for pair in my_pairs:
        for word in pair[1].split(" "):
            if word and word not in french_word2index:
                french_word2index[word] = french_word_n
                # 更新下一个单词的下标值
                french_word_n += 1
    # english_index2word  french_index2word
    english_index2word = {v : k for k, v in english_word2index.items()}
    french_index2word = {v : k for k, v in french_word2index.items()}

    print("len(english_word2index)-->", len(english_word2index))
    print("len(french_word2index)-->", len(french_word2index))
    print("english_word_n--->", english_word_n, "french_word_n-->", french_word_n)

    return (
        english_word2index,
        english_index2word,
        english_word_n,
        french_word2index,
        french_index2word,
        french_word_n,
        my_pairs
    )

# -------------------------------------------------步骤二:构建数据源对象-------------------------------------------------

# 步骤二: 构建数据源对象
# 原始数据 -> 数据源MyPairsDataset --> 数据迭代器DataLoader
# 构造数据源 MyPairsDataset，把语料xy 文本数值化 再转成tensor_x tensor_y
class MyPairsDataset(Dataset):
    def __init__(self, my_pairs, english_word2index, french_word2index):
        self.my_pairs = my_pairs
        self.english_word2index = english_word2index
        self.french_word2index = french_word2index
        # 样本条目
        self.sample_len = len(my_pairs)

    # 获取样本条数
    def __len__(self):
        return self.sample_len

    # 获取第几条样本数据
    def __getitem__(self, index):
        # 对index异常值进行修正[0, self.sample_len-1]
        index = min(max(index, 0), self.sample_len -1)

        # 按索引获数据样本x y
        x = self.my_pairs[index][0] # 英语句子
        y = self.my_pairs[index][1] # 法语句子

        # 样本x 文本数值化
        x_indices = [self.english_word2index[word] for word in x.split(" ") if word in self.english_word2index]
        x_indices.append(EOS_token)
        tensor_x = torch.tensor(x_indices, dtype=torch.long, device=device)
        print("tensor_x.shape--> ", tensor_x.shape)
        # 样本y 文本数值化
        y_indices = [self.french_word2index[word] for word in y.split(" ") if word in self.french_word2index]
        y_indices.append(EOS_token)
        tensor_y = torch.tensor(y_indices, dtype=torch.long, device=device)
        print("tensor_y.shape", tensor_y.shape)
        # 注: tensor_x, tensor_y 都是一维数组, 通过DataLoader拿出来的数据是二维数据
        return tensor_x, tensor_y

def dm_test_MyPairsDataset():
    # 1 调用my_getdata函数获取数据
    (
        english_word2index,
        english_index2word,
        english_word_n,
        french_word2index,
        french_index2word,
        french_word_n,
        my_pairs,
    ) = getdata()

    # 2 实例化dataset对象
    mypairsdataset = MyPairsDataset(my_pairs, english_word2index, french_word2index)

    # 3 实例化dataloader
    mydataloader = DataLoader(dataset=mypairsdataset, batch_size=1, shuffle=True)
    for i, (x, y) in enumerate(mydataloader):
        print("x.shape", x.shape, x)
        print("y.shape", y.shape, y)
        break

# -------------------------------------------------步骤三:构建基于GRU的编码器-------------------------------------------------
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        # input_size: 编码器 词嵌入层单词数
        # hidden_size: 编码器 词嵌入层隐藏维度数(单词的特征树)
        super(EncoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # 实例化embedding层
        self.embedding = nn.Embedding(
            num_embeddings=self.input_size, embedding_dim=self.hidden_size
        )
        # 实例化nn.GRU层 参数batch_first=True
        self.gru = nn.GRU(
            input_size=self.hidden_size, hidden_size=self.hidden_size,batch_first=True
        )

    # 前向传播
    def forward(self, input, hidden):
        # 传入经过词嵌入层的数据
        output = self.embedding(input)

        output, hidden = self.gru(output, hidden)
        return output, hidden
    # 显示初始化参数
    def inithidden(self):
        # 将隐藏层张量初始化成为1x1xself.hidden_size大小的张量
        return torch.zeros(size=(1, 1, self.hidden_size), device=device)

def dm_test_EncoderRNN():
    # 调用my_getdata函数获取数据
    (
        english_word2index,
        english_index2word,
        english_word_n,
        french_word2index,
        french_index2word,
        french_word_n,
        my_pairs,
    ) = getdata()
    # 实例化dataset对象
    mypairsdataset = MyPairsDataset(my_pairs, english_word2index, french_word2index)

    # 实例化dataloader
    mydataloader = DataLoader(dataset=mypairsdataset, batch_size=1, shuffle=True)

    # 实例化模型
    input_size = english_word_n
    hidden_size = 256
    my_encoderrnn = EncoderRNN(input_size, hidden_size).to(device=device)
    print("my_encoderrnn模型结构--->", my_encoderrnn)

    # 给encode模型喂数据
    for i, (x, y) in enumerate(mydataloader):
        print("x.shape", x.shape, x)
        print("y.shape", y.shape, y)

        # 一次性的送数据
        hidden = my_encoderrnn.inithidden()
        # encode_output_c: 未加attention的中间语义张量c
        encode_output_c, hidden = my_encoderrnn(x, hidden)
        print("encode_output_c.shape--->", encode_output_c.shape, encode_output_c)
        break
# -------------------------------------------------步骤四:构建基于GRU和Attention的解码器-------------------------------------------------
class GRUAttnDecoderRNN(nn.Module):
    # init 定义6个层
    def __init__(self, output_size, hidden_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(GRUAttnDecoderRNN, self).__init__()
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        # 定义解码器embedding层
        self.embedding = nn.Embedding(num_embeddings=self.output_size, embedding_dim=self.hidden_size)

        # 定义线性层1,求q的注意力权重分布
        # 查询张量Q:解码器每个时间步的隐藏层输出或者是当前输入x
        # 键张量k:解码器上一个时间步的隐藏层输出
        self.attn = nn.Linear(in_features=self.hidden_size*2, out_features=self.max_length)

        # 定义线性层2,q+注意力结果表示融合后,再按照指定维度输出
        # 值张量v:编码器部分每个时间步输出结果组合而成
        self.attn_combine = nn.Linear(in_features=self.hidden_size*2, out_features=self.hidden_size)

        # 定义dropout层
        self.dropout = nn.Dropout(p=self.dropout_p)

        # 定义GRU层
        self.gru = nn.GRU(input_size=self.hidden_size, hidden_size=self.hidden_size, batch_first=True)

        # 定义output层
        self.out = nn.Linear(in_features=self.hidden_size, out_features=self.output_size)

        # 实例化softmax,归一化,便于分类
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden, encoder_outputs):
        # 数据经过词嵌入
        embedded = self.embedding(input)

        # 使用dropout,防止过拟合
        embedded = self.dropout(embedded)

        # 1. 求查询张量q的注意力权重分布
        attn_weights = torch.softmax(
            self.attn(torch.cat(tensors=(embedded,hidden),dim=-1)), dim=-1
        )
        # 2. 求查询张量q的注意力表示 bmm运算
        attn_applied = torch.bmm(input=attn_weights, mat2=encoder_outputs)
        # 3. q与attn_applied融合
        output = torch.cat(tensors=(embedded, attn_applied), dim=-1)
        # 再按照指定维度 output,gru层输入形状要求
        output = self.attn_combine(output)

        # 4. 查询张量q的注意力结果表示,使用ReLU函数激活
        output = torch.relu(output)

        # 查询张量经过gru、softmax进行分类,输出
        output, hidden = self.gru(output, hidden)

        # output 经过全连接层 out+softmax, 全连接层要求输入为二维数据
        output = self.softmax(self.out(output[:, 0, :]))

        # 返回解码器分类,最后隐藏状态, 注意力权重
        return output, hidden, attn_weights


# -------------------------------------------------步骤五:构建内部迭代训练函数-------------------------------------------------

def train_iters(
        x,
        y,
        my_encoderrnn: EncoderRNN,
        my_attndecoderrnn: GRUAttnDecoderRNN,
        myadam_encode,
        myadam_decode,
        mynllloss,
        total_steps,
        current_syep,
):
    my_encoderrnn.train()
    my_attndecoderrnn.train()
    # 1. 编码
    encode_hidden = my_encoderrnn.inithidden()
    encode_output, encode_hidden = my_encoderrnn(x, encode_hidden)

    # 2. 解码参数准备和解码
    encode_output_c = torch.zeros(
        1, MAX_LENGTH, my_encoderrnn.hidden_size, device=device
    )
    for idx in range(x.shape[1]):
        encode_output_c[:, idx, :] = encode_output[:, idx, :]

    # 解码参数2
    decode_hidden = encode_hidden
    # 解码参数3
    input_y = torch.tensor([[SOS_token]], device=device)

    myloss = 0.0
    iters_num = 0
    y_len = y.shape[1]
    # 教师强制机制, 阈值线性衰减
    teacher_forcing_ratio = max(0.1, 1-(current_syep/total_steps))
    # 阈值指数衰减
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    for idx in range(y_len):
        output_y, decoder_hidden, attn_weight = my_attndecoderrnn(
            input_y, decode_hidden, encode_output_c
        )
        target_y = y[:, idx]
        myloss = myloss + mynllloss(output_y, target_y)
        iters_num += 1
        # 使用teacher_forcing
        if use_teacher_forcing:
            # 获取真实样本作为下一个输入
            input_y = y[:, idx].reshape(shape=(-1, 1))
        # 不使用teacher_forcing
        else:
            # 获取最大值和索引
            topv, topi = output_y.topk(1)
            if topi.item() == EOS_token:
                break
            # 获取预测y值作为下一个输入
            input_y = topi.detach()

    # 梯度清零
    myadam_encode.zero_grad()
    myadam_decode.zero_grad()

    # 反向传播
    myloss.backward()

    # 梯度更新
    myadam_encode.step()
    myadam_decode.step()

    # 计算迭代次数的平均损失
    return myloss.item() / iters_num

# -------------------------------------------------步骤六:构建模型训练函数-------------------------------------------------
def train_seq2seq():
    # 获取数据
    (english_word2index, english_index2word, english_word_n,
     french_word2index, french_index2word, french_word_n, my_pairs) = getdata()
    # 实例化 mypairsdataset对象  实例化 mydataloader
    mypairsdataset = MyPairsDataset(my_pairs, english_word2index, french_word2index)
    mydataloader = DataLoader(dataset=mypairsdataset, batch_size=1, shuffle=True)

    # 实例化编码器和编码器优化器
    my_encoderrnn = EncoderRNN(english_word_n, 256).to(device)
    my_attndecoderrnn = GRUAttnDecoderRNN(output_size=french_word_n, hidden_size=256, dropout_p=0.1, max_length=10).to(device)

    myadam_encode = optim.Adam(my_encoderrnn.parameters(), lr=mylr)
    myadam_decode = optim.Adam(my_attndecoderrnn.parameters(), lr=mylr)

    # 实例化损失函数
    mynllloss = nn.NLLLoss()

    # 定义模型训练的承诺书
    plot_loss_list = []
    # 统计所有轮次的总批次
    total_steps = epochs * len(mydataloader)
    # 当前累积批次数
    current_step = 0

    # 外层循环:控制轮数
    for epoch_idx in range(1, epochs+1):
        print_loss_total, plot_loss_total = 0.0, 0.0
        starttime = time.time()

        # 内层循环:控制迭代次数
        for item, (x, y) in enumerate(mydataloader, start=1):
            # 调用内部训练函数
            myloss = train_iters(x, y, my_encoderrnn, my_attndecoderrnn, myadam_encode, myadam_decode, mynllloss, total_steps, current_step)
            print_loss_total += myloss
            plot_loss_total += myloss
            # 累计训练批次数
            current_step += 1

            # 计算打印屏幕间隔损失-每隔1000次
            if item % print_interval_num == 0:
                print_loss_avg = print_loss_total / print_interval_num
                # 将总损失归0
                print_loss_total = 0
                # 打印日志:训练耗时、当前迭代步、当前进度百分比、当前平均损失
                print('轮次%d  损失%.6f 时间:%d' % (epoch_idx, print_loss_avg, time.time() - starttime))
            # 计算画图间隔损失-每隔100次
            if item % plot_interval_num == 0:
                # 通过总损失除以间隔得到平均损失
                plot_loss_avg = plot_loss_total / plot_interval_num
                # 将平均损失添加plot_loss_list列表中
                plot_loss_list.append(plot_loss_avg)
                # 总损失归0
                plot_loss_total = 0
        # 每个轮次保存模型
        torch.save(my_encoderrnn.state_dict(), 'model/my_encoderrnn_%d.pth' % epoch_idx)
        torch.save(my_attndecoderrnn.state_dict(), 'model/my_attndecoderrnn_%d.pth' % epoch_idx)

    # 所有轮次训练完毕 画损失图
    plt.figure()
    plt.plot(plot_loss_list)
    plt.savefig('img/s2sq_loss.png')
    plt.show()

# ------------------------------------------------ 步骤七:构建模型评估函数 ------------------------------------------------
# 模型评估代码与模型预测代码类似，需要注意使用with torch.no_grad()
# 模型预测时，第一个时间步使用SOS_token作为输入 后续时间步采用预测值作为输入，也就是自回归机制
def seq2seq_evaluate(
    x, my_encoderrnn: EncoderRNN, my_attndecoderrnn: GRUAttnDecoderRNN, french_index2word
):
    with torch.no_grad():
        my_encoderrnn.eval()
        my_attndecoderrnn.eval()
        # 1 编码：一次性的送数据
        encode_hidden = my_encoderrnn.inithidden()
        encode_output, encode_hidden = my_encoderrnn(x, encode_hidden)

        # 2 解码参数准备
        # 解码参数1 固定长度中间语义张量c
        encoder_outputs_c = torch.zeros(
            1, MAX_LENGTH, my_encoderrnn.hidden_size, device=device
        )
        x_len = x.shape[1]
        for idx in range(x_len):
            encoder_outputs_c[:, idx, :] = encode_output[:, idx, :]

        # 解码参数2 最后1个隐藏层的输出 作为 解码器的第1个时间步隐藏层输入
        decode_hidden = encode_hidden

        # 解码参数3 解码器第一个时间步起始符
        input_y = torch.tensor([[SOS_token]], device=device)

        # 3 自回归方式解码
        # 初始化预测的词汇列表
        decoded_words = []
        # 初始化attention张量
        decoder_attentions = torch.zeros(1, MAX_LENGTH, MAX_LENGTH)
        for idx in range(MAX_LENGTH):  # note:MAX_LENGTH=10
            output_y, decode_hidden, attn_weights = my_attndecoderrnn(
                input_y, decode_hidden, encoder_outputs_c
            )
            # 预测值作为下一次时间步的输入值
            topv, topi = output_y.topk(1)
            decoder_attentions[:, idx, :] = attn_weights[:, 0, :]

            # 如果输出值是终止符，则循环停止
            if topi.item() == EOS_token:
                decoded_words.append("<EOS>")
                break
            else:
                decoded_words.append(french_index2word[topi.item()])

            # 将本次预测的索引赋值给 input_y，进行下一个时间步预测
            input_y = topi.detach()

    # 返回结果decoded_words，注意力张量权重分布表(把没有用到的部分切掉)
    # 句子长度最大是10, 长度不为10的句子的注意力张量其余位置为0, 去掉
    return decoded_words, decoder_attentions[:, :idx + 1, :]

PATH1 = "model/my_encoderrnn_2.pth"
PATH2 = "model/my_attndecoderrnn_2.pth"

def dm_test_seq2seq_evaluate():
    (
        english_word2index,
        english_index2word,
        english_word_n,
        french_word2index,
        french_index2word,
        french_word_n,
        my_pairs,
    ) = getdata()
    # 实例化dataset对象
    mypairsdataset = MyPairsDataset(my_pairs, english_word2index, french_word2index)

    # 实例化模型
    input_size = english_word_n
    hidden_size = 256  # 观察结果数据 可使用8
    my_encoderrnn = EncoderRNN(input_size, hidden_size).to(device)

    """
    torch.load(map_location=)
    map_location: 指定如何重映射模型权重的存储设备（如 GPU → CPU 或 GPU → 其他 GPU）。
    # 加载到 CPU：map_location=torch.device('cpu') 或 map_location='cpu'。
    自动选择可用设备：map_location=torch.device('cuda')。
    自定义映射逻辑：通过函数定义设备映射规则。
    map_location=lambda storage, loc: storage -> 该lambda函数直接返回原始存储对象(storage)
    强制所有张量保留在保存时的设备上。当模型权重保存时的设备与当前环境一致时（例如均在CPU或同一GPU上），避免不必要的设备迁移。

    load_state_dict(strict=)
    strict:True（默认）:要求加载的权重键（keys）与当前模型的键完全匹配。如果存在不匹配（例如权重中缺少某些键，或模型有额外键），抛出RuntimeError。
    """
    my_encoderrnn.load_state_dict(
        torch.load(PATH1, map_location=lambda storage, loc: storage), strict=False
    )
    print("my_encoderrnn模型结构--->", my_encoderrnn)

    # 实例化模型
    input_size = french_word_n
    hidden_size = 256  # 观察结果数据 可使用8
    my_attndecoderrnn = GRUAttnDecoderRNN(input_size, hidden_size).to(device)
    # my_attndecoderrnn.load_state_dict(torch.load(PATH2))
    my_attndecoderrnn.load_state_dict(
        torch.load(PATH2, map_location=lambda storage, loc: storage), False
    )
    print("my_decoderrnn模型结构--->", my_attndecoderrnn)

    my_samplepairs = [
        [
            "i m impressed with your french .",
            "je suis impressionne par votre francais .",
        ],
        ["i m more than a friend .", "je suis plus qu une amie ."],
        ["she is beautiful like her mother .", "elle est belle comme sa mere ."],
    ]
    print("my_samplepairs--->", len(my_samplepairs))

    for index, pair in enumerate(my_samplepairs):
        x = pair[0]
        y = pair[1]

        # 对输入句子进行与训练数据相同的预处理
        x = normalizeString(x)

        # 样本x 文本数值化
        tmpx = []
        for word in x.split(" "):
            # 检查词汇是否在词汇表中，如果不在则跳过
            if word in english_word2index:
                tmpx.append(english_word2index[word])
            else:
                print(f"警告: 词汇 '{word}' 不在词汇表中，已跳过")

        if not tmpx:  # 如果没有有效词汇，添加一个默认词（如SOS）
            tmpx.append(SOS_token)

        tmpx.append(EOS_token)
        tensor_x = torch.tensor(tmpx, dtype=torch.long, device=device).view(1, -1)

        # 模型预测
        decoded_words, attentions = seq2seq_evaluate(
            tensor_x, my_encoderrnn, my_attndecoderrnn, french_index2word
        )
        # print("attentions--->", attentions)
        # print('decoded_words->', decoded_words)
        output_sentence = " ".join(decoded_words)

        print("\n")
        print(">", x)
        print("=", y)
        print("<", output_sentence)

# ------------------------------------------------ 步骤八:Attention张量制图 ------------------------------------------------
def dm_test_Attention():
    (
        english_word2index,
        english_index2word,
        english_word_n,
        french_word2index,
        french_index2word,
        french_word_n,
        my_pairs,
    ) = getdata()

    # 实例化dataset对象
    mypairsdataset = MyPairsDataset(my_pairs, english_word2index, french_word2index)
    # 实例化dataloader
    mydataloader = DataLoader(dataset=mypairsdataset, batch_size=1, shuffle=True)

    # 实例化模型
    input_size = english_word_n
    hidden_size = 256  # 观察结果数据 可使用8
    my_encoderrnn = EncoderRNN(input_size, hidden_size).to(device=device)
    # my_encoderrnn.load_state_dict(torch.load(PATH1))
    my_encoderrnn.load_state_dict(
        torch.load(PATH1, map_location=lambda storage, loc: storage), False
    )

    # 实例化模型
    input_size = french_word_n
    hidden_size = 256  # 观察结果数据 可使用8
    my_attndecoderrnn = GRUAttnDecoderRNN(input_size, hidden_size).to(device=device)
    # my_attndecoderrnn.load_state_dict(torch.load(PATH2))
    my_attndecoderrnn.load_state_dict(
        torch.load(PATH2, map_location=lambda storage, loc: storage), False
    )

    sentence = "we re both teachers ."
    # 对句子进行预处理，确保与训练数据处理方式一致
    sentence = normalizeString(sentence)

    # 样本x 文本数值化，添加词汇检查
    tmpx = []
    for word in sentence.split(" "):
        # 检查词汇是否在词汇表中，如果不在则跳过
        if word in english_word2index:
            tmpx.append(english_word2index[word])
        else:
            print(f"警告: 词汇 '{word}' 不在词汇表中，已跳过")

    if not tmpx:  # 如果没有有效词汇，添加一个默认词
        tmpx.append(SOS_token)

    # 样本x 文本数值化
    # tmpx = [english_word2index[word] for word in sentence.split(" ")]
    tmpx.append(EOS_token)
    tensor_x = torch.tensor(tmpx, dtype=torch.long, device=device).view(1, -1)

    # 模型预测
    decoded_words, attentions = seq2seq_evaluate(
        tensor_x, my_encoderrnn, my_attndecoderrnn, french_index2word
    )
    print("decoded_words->", decoded_words)

    # print('\n')
    # print('英文', sentence)
    # print('法文', output_sentence)

    # 创建热图
    fig, ax = plt.subplots(figsize=(10, 8))
    # cmap:指定一个颜色映射，将数据值映射到颜色
    # viridis:从深紫色（低值）过渡到黄色（高值），具有良好的对比度和可读性
    cax = ax.matshow(attentions[0].cpu().detach().numpy(), cmap="viridis")
    # 添加颜色条
    fig.colorbar(cax)

    # 设置坐标轴标签
    ax.set_xticks(range(len(sentence.split())))
    ax.set_xticklabels(sentence.split(), rotation=45)
    ax.set_yticks(range(len(decoded_words)))
    ax.set_yticklabels(decoded_words)

    # 添加标签
    for (i, j), value in np.ndenumerate(attentions[0].cpu().detach().numpy()):
        ax.text(j, i, f"{value:.2f}", ha="center", va="center", color="white")
    # 保存图像
    plt.savefig("img/s2s_attn.png")
    plt.show()

    print("attentions.numpy()--->\n", attentions.numpy())
    print("attentions.size--->", attentions.size())

# ------------------------------------------------ main ------------------------------------------------
if __name__ == '__main__':
    # dm_test_MyPairsDataset()
    # dm_test_EncoderRNN()
    # train_seq2seq()
    # dm_test_seq2seq_evaluate()
    dm_test_Attention()