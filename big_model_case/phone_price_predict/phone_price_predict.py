# Created by erainm on 2025/8/16 11:10.
# IDE：PyCharm 
# @Project: ai_big_model
# @File：phone_price_predict
# @Description: 手机价格预测案例
# TODO: 基础版

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import torch
import time

"""
    获取数据并处理数据
"""
def get_data_and_processing(data_url, batch_size):
    # 加载数据
    data = pd.read_csv(data_url)
    # 洞察数据
    # print("数据形状 --> ", data.shape) # 2000行，21列
    # print("数据列 --> ", data.columns)
    # print("数据信息 --> ", data.info())
    # print("数据前几条: --> ", data.head())
    # 数据处理
    # 获取特征值x(1~20列)，目标值y(21列)
    x = data.iloc[:, :-1]
    y = data.iloc[:,-1]
    # 3.2为了后续用于张量,提前做类型转换：特征值转浮点，目标值转整型
    x = x.astype(np.float32)
    y = y.astype(np.int64)
    # 3.3 数据集划分
    x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.2, random_state=88)
    # 3.4 todo 构建数据集,最终为训练集dataloader和测试集dataloader
    # 先把numpy数据集转换成张量,然后封装成张量数据集
    train_dataset = TensorDataset(torch.from_numpy(x_train.values), torch.from_numpy(y_train.values))
    valid_dataset = TensorDataset(torch.from_numpy(x_valid.values), torch.from_numpy(y_valid.values))
    # 再把张量数据集封装成数据加载器,并且设置批量大小和是否打乱数据
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    # 4.返回结果: 数据加载器,数据集长度,输入特征维度,输出特征维度len(y.unique()0,1,2,3
    return train_dataloader, valid_dataloader, x_train.shape[1], len(y.unique())

# 手机价格预测模型
class PhonePricePredictModel(torch.nn.Module):
    # 重写__init__方法和 forward方法
    def __init__(self, input_num, output_num):
        # 调用父类init方法
        super(PhonePricePredictModel, self).__init__()
        # 定义网络结构, 神经网络隐藏层
        self.linear1 = torch.nn.Linear(input_num, 256)
        self.linear2 = torch.nn.Linear(256, 512)
        self.out = torch.nn.Linear(512, output_num)
    # 重写前向传播
    def forward(self, x):
        # 加权求和, 激活函数(隐藏层使用ReLU作为激活函数)
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))

        # 输出层,返回加权求的结果
        out = self.out(x)
        return out

# 模型训练
def train_model(train_dataloader, phone_price_predict_model, epochs, model_url):
    # 训练数据以及训练使用的模型,参数已经传入
    # 创建多分类损失函数对象
    loss_fn = torch.nn.CrossEntropyLoss()
    # 创建优化器对象
    optimizer = torch.optim.SGD(phone_price_predict_model.parameters(), lr=0.001)
    # 循环训练模型(外层循环:轮次,内层循环:批次)
    for epoch in range(epochs):
        total_loss, batch_cnt, start = 0.00, 0, time.time()
        for batch_x, batch_y in train_dataloader:
            # 前向传播:输入到输出, 预测值和损失值
            # 模型预测
            y_predict = phone_price_predict_model(batch_x)
            # 损失计算
            loss = loss_fn(y_predict, batch_y)
            # 累加损失值和批次
            total_loss += loss
            batch_cnt += 1
            # 反向传播,从输出到输入:梯度计算和参数更新
            # 先进行梯度清零
            optimizer.zero_grad()
            # 梯度自动计算
            loss.backward()
            # 参数更新
            optimizer.step()
        epoch_loss = total_loss / batch_cnt
        print(f"第{epoch + 1}轮,运行时间{time.time() - start:.2f}秒,损失值为:{epoch_loss:.2f}")
    torch.save(phone_price_predict_model.state_dict(), model_url)

# 模型评估
def model_eval(valid_dataloader, input_num, output_num, model_url):
    # 创建模型对象
    phone_price_predict_test_model = PhonePricePredictModel(input_num, output_num)
    # 使用测试集测试训练好的模型
    phone_price_predict_test_model.load_state_dict(torch.load(model_url))
    # 定义预测正确样本数
    correct = 0
    total = 0
    # 每批次评估
    for batch_x, batch_y in valid_dataloader:
        # 模型预测
        y = phone_price_predict_test_model(batch_x)
        # 最大的获取预测结果
        y_pred = torch.argmax(y, dim=1)
        # 获取预测正确的个数
        correct += (y_pred == batch_y).sum().item()
        total += batch_y.size(0)
    print(f'Acc: {(correct / total):.4f}')

if __name__ == '__main__':
    data_url = "手机价格预测.csv"
    model_url = "model/phone_price_predict_model.pth"
    batch_size = 8
    try:
        # 1. 加载并处理数据
        train_dataloader, valid_dataloader, input_num, output_num = get_data_and_processing(data_url=data_url, batch_size=batch_size)

        # 2. 创建类，构造神经网络
        phone_price_predict_model = PhonePricePredictModel(input_num, output_num)
        # 3. 训练模型
        epochs = 3
        train_model(train_dataloader, phone_price_predict_model, epochs, model_url)
        # 4. 模型评估
        model_eval(valid_dataloader, input_num, output_num, model_url)
    except Exception as e:
        print(f"执行过程中出现错误: {e}")
        import  traceback
        traceback.print_exc()