# Created by erainm on 2025/8/24 11:12.
# IDE：PyCharm 
# @Project: ai_big_model
# @File：gru_model
# @Description: gru模型示例
# TODO:


import torch
import torch.nn as nn


class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        """
        GRU模型初始化

        参数:
        input_size: 输入特征维度
        hidden_size: 隐藏层维度
        num_layers: GRU层数
        output_size: 输出维度
        """
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 定义GRU层
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True  # 输入格式为(batch_size, sequence_length, input_size)
        )

        # 定义全连接输出层
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        前向传播

        参数:
        x: 输入张量，形状为(batch_size, sequence_length, input_size)

        返回:
        output: 输出张量，形状为(batch_size, output_size)
        """
        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # GRU前向传播
        # out: (batch_size, sequence_length, hidden_size)
        # hn: (num_layers, batch_size, hidden_size)
        out, _ = self.gru(x, h0)

        # 只取最后一个时间步的输出
        # out: (batch_size, hidden_size)
        out = out[:, -1, :]

        # 通过全连接层得到最终输出
        # output: (batch_size, output_size)
        output = self.fc(out)

        return output


# 示例用法
if __name__ == "__main__":
    # 定义模型参数
    batch_size = 32
    sequence_length = 10
    input_size = 5
    hidden_size = 64
    num_layers = 2
    output_size = 1

    # 创建模型实例
    model = GRUModel(input_size, hidden_size, num_layers, output_size)

    # 创建随机输入数据
    x = torch.randn(batch_size, sequence_length, input_size)

    # 前向传播
    output = model(x)

    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"模型结构:\n{model}")