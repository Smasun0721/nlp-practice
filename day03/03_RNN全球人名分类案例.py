import json
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import string
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import json

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 获取所有常见字符
all_letters = string.ascii_letters + " .,;'"
# 常见字符数量
letters_len = len(all_letters)

# 获取所有国家列表
categories = [
    "Italian",
    "English",
    "Arabic",
    "Spanish",
    "Scottish",
    "Irish",
    "Chinese",
    "Vietnamese",
    "Japanese",
    "French",
    "Greek",
    "Dutch",
    "Korean",
    "Polish",
    "Portuguese",
    "Russian",
    "Czech",
    "German",
]
# 国家数量
categories_num = len(categories)


# 读取数据
def read_data(file_path):
    """
    读取文件到内存中，返回 特征列表 与 标签列表
    :param file_path: 源文件路径
    :return: list_x: 特征列表   list_y: 标签列表
    """
    x_list = []
    y_list = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            # 过滤无效数据，行长度<=5
            if len(line.strip()) <= 5:
                continue
            x, y = line.strip().split("\t")
            x_list.append(x)
            y_list.append(y)

    return x_list, y_list


# 人名分类数据集
class NameClassDataset(Dataset):
    def __init__(self, x_list, y_list):
        self.x_list = x_list
        self.y_list = y_list
        self.samples_len = len(x_list)

    def __len__(self):
        return self.samples_len

    def __getitem__(self, index):
        index = min(max(index, 0), self.samples_len - 1)
        x = self.x_list[index]
        y = self.y_list[index]
        # x 的 one-hot编码
        tensor_x = torch.zeros(len(x), letters_len)
        for i, letter in enumerate(x):
            letter_index = all_letters.find(letter)
            tensor_x[i][letter_index] = 1
        # y 在国家列表中的索引
        torch_y = torch.tensor(categories.index(y), dtype=torch.long)
        # 返回数据
        return tensor_x, torch_y


# 获取数据加载器
def get_dataloader(x_list, y_list, batch_size):
    dataset = NameClassDataset(x_list, y_list)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


# 搭建神经网络模型
# RNN
class NameClassRNNModel(nn.Module):
    # 初始化
    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        # rnn层
        self.rnn = nn.RNN(self.input_size, self.hidden_size, self.n_layers)
        # 全连接层
        self.linear = nn.Linear(self.hidden_size, self.output_size)

    # 前向传播
    def forward(self, input, hidden):
        # input -> [seq_len, batch_size, input_size]
        # hidden -> [n_layers, batch_size, hidden_size]
        output, hidden = self.rnn(input, hidden)  # RNN层
        output = self.linear(output[-1])  # 获取最顶层隐藏层接入全连接层
        return output, hidden  # 返回数据

    # 初始化隐藏层
    def init_hidden(self, batch_size=1):
        return torch.zeros(self.n_layers, batch_size, self.hidden_size)


# LSTM
class NameClassLSTMModel(nn.Module):
    # 初始化
    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        # LSTM层
        self.rnn = nn.LSTM(self.input_size, self.hidden_size, self.n_layers)
        # 全连接层
        self.linear = nn.Linear(self.hidden_size, self.output_size)

    # 前向传播
    def forward(self, input, hidden, cell):
        # input -> [seq_len, batch_size, input_size]
        # hidden -> [n_layers, batch_size, hidden_size]
        output, (hidden, cell) = self.rnn(input, (hidden, cell))  # RNN层
        output = self.linear(output[-1])  # 获取最顶层隐藏层接入全连接层
        return output, hidden, cell  # 返回数据

    # 初始化隐藏层
    def init_hidden_and_cell(self, batch_size=1):
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_size)
        cell = torch.zeros(self.n_layers, batch_size, self.hidden_size)
        return hidden, cell


# GRU
class NameClassGRUModel(nn.Module):
    # 初始化
    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        # GRU层
        self.rnn = nn.GRU(self.input_size, self.hidden_size, self.n_layers)
        # 全连接层
        self.linear = nn.Linear(self.hidden_size, self.output_size)

    # 前向传播
    def forward(self, input, hidden):
        # input -> [seq_len, batch_size, input_size]
        # hidden -> [n_layers, batch_size, hidden_size]
        output, hidden = self.rnn(input, hidden)  # RNN层
        output = self.linear(output[-1])  # 获取最顶层隐藏层接入全连接层
        return output, hidden  # 返回数据

    # 初始化隐藏层
    def init_hidden(self, batch_size=1):
        return torch.zeros(self.n_layers, batch_size, self.hidden_size)


def rnn_test():
    rnn = NameClassRNNModel(
        input_size=letters_len, hidden_size=128, output_size=categories_num
    )
    print(rnn)

    input = torch.randn(6, letters_len)
    input.unsqueeze_(1)  # 增加 batch_size 维度，默认为 1
    h0 = rnn.init_hidden()
    output, hn = rnn(input, h0)
    print(f"rnn_output.shape: {output.shape}")
    print(f"rnn_hn.shape: {hn.shape}")


def lstm_test():
    lstm = NameClassLSTMModel(
        input_size=letters_len, hidden_size=128, output_size=categories_num
    )
    print(lstm)

    input = torch.randn(6, letters_len)
    input.unsqueeze_(1)  # 增加 batch_size 维度，默认为 1
    h0, cell = lstm.init_hidden_and_cell()
    output, hn, cell = lstm(input, h0, cell)
    print(f"lstm_output.shape: {output.shape}")
    print(f"lstm_hn.shape: {hn.shape}")
    print(f"lstm_cell.shape: {cell.shape}")


def gru_test():
    gru = NameClassGRUModel(
        input_size=letters_len, hidden_size=128, output_size=categories_num
    )
    print(gru)

    input = torch.randn(6, letters_len)
    input.unsqueeze_(1)  # 增加 batch_size 维度，默认为 1
    h0 = gru.init_hidden()
    output, hn = gru(input, h0)
    print(f"gru_output.shape: {output.shape}")
    print(f"gru_hn.shape: {hn.shape}")


import os


def train_one_model(
    model, model_name, dataloader, epochs=10, lr=0.001, save_path: str = None
):
    """
    通用训练函数
    :param model: 模型实例（RNN/LSTM/GRU）
    :param model_name: 模型名称，用于打印和画图
    :param dataloader: 数据加载器
    :param epochs: 训练轮数
    :param lr: 学习率
    :param save_path: 模型保存路径，None 则不保存
    :return: 每个 epoch 的平均 loss 列表
    """
    # 1.训练前准备
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # 判断是否是 LSTM（LSTM 需要 hidden 和 cell 两个状态）
    is_lstm = isinstance(model, NameClassLSTMModel)

    loss_list = []
    acc_list = []
    start_time = time.time()
    print(f"开始训练 {model_name}...")

    # 2.模型训练
    for epoch in range(epochs):
        total_loss = 0  # 该轮总损失
        total_correct = 0  # 该轮正确个数
        for i, (x, y) in enumerate(tqdm(dataloader)):  # 加入进度条
            # x 的形状为 [batch_size, seq_len, input_size]
            x = x.transpose(0, 1)  # -> [seq_len, batch_size, input_size]

            # 初始化隐藏层 + 前向传播
            if is_lstm:
                hidden, cell = model.init_hidden_and_cell()
                output, hidden, cell = model(x, hidden, cell)
            else:
                hidden = model.init_hidden()
                output, hidden = model(x, hidden)

            # 损失计算
            loss = criterion(output, y)

            # 梯度清零 + 反向传播 + 参数更新
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 统计
            total_loss += loss.item()
            total_correct += (output.argmax(1) == y).sum().item()

        # 统计
        mean_loss = total_loss / len(dataloader)
        loss_list.append(mean_loss)
        acc_list.append(total_correct / len(dataloader))

        # 打印进度
        print(
            f"epoch: {epoch+1}/{epochs}   {model_name}_loss: {mean_loss:.4f}    {model_name}_acc: {acc_list[-1]:.4f}   time: {time.time() - start_time:.2f}s"
        )

    # 3.保存模型
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model, save_path)

    return loss_list, acc_list


def dm_train_rnn():
    x_list, y_list = read_data("./data/name_classfication.txt")
    dataloader = get_dataloader(x_list, y_list, batch_size=1)
    model = NameClassRNNModel(letters_len, 128, categories_num)
    loss_list, acc_list = train_one_model(
        model, "RNN", dataloader, epochs=10, lr=0.001, save_path="./model/rnn.bin"
    )
    return loss_list, acc_list


def dm_train_lstm():
    x_list, y_list = read_data("./data/name_classfication.txt")
    dataloader = get_dataloader(x_list, y_list, batch_size=1)
    model = NameClassLSTMModel(letters_len, 128, categories_num)
    loss_list, acc_list = train_one_model(
        model, "LSTM", dataloader, epochs=10, lr=0.001, save_path="./model/lstm.bin"
    )
    return loss_list, acc_list


def dm_train_gru():
    x_list, y_list = read_data("./data/name_classfication.txt")
    dataloader = get_dataloader(x_list, y_list, batch_size=1)
    model = NameClassGRUModel(letters_len, 128, categories_num)
    loss_list, acc_list = train_one_model(
        model, "GRU", dataloader, epochs=10, lr=0.001, save_path="./model/gru.bin"
    )
    return loss_list, acc_list


def compare():
    # 读取保存的数据
    try:
        with open("./model/training_results.json", "r", encoding="utf-8") as f:
            results = json.load(f)
    except FileNotFoundError:
        print("未找到训练结果文件，请先运行训练！")
        return

    # 创建一个 1 行 2 列的画布
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 画 Loss 对比图
    axes[0].plot(results["RNN"]["loss"], label="RNN Loss")
    axes[0].plot(results["LSTM"]["loss"], label="LSTM Loss")
    axes[0].plot(results["GRU"]["loss"], label="GRU Loss")
    axes[0].set_title("模型训练 Loss 对比")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    # 画 Acc 对比图
    axes[1].plot(results["RNN"]["acc"], label="RNN Acc")
    axes[1].plot(results["LSTM"]["acc"], label="LSTM Acc")
    axes[1].plot(results["GRU"]["acc"], label="GRU Acc")
    axes[1].set_title("模型训练 Accuracy 对比")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    plt.tight_layout()
    plt.show()


# 模型预测
rnn_path = "./model/rnn.bin"
lstm_path = "./model/lstm.bin"
gru_path = "./model/gru.bin"


# 将 输入文本 转换成 one-hot编码
def lineToTensor(x):
    tensor_x = torch.zeros(len(x), letters_len)
    for i, c in enumerate(x):
        tensor_x[i][all_letters.find(c)] = 1
    return tensor_x


def predict(x, model_param_path):
    # 定义变量
    input_size = letters_len
    hidden_size = 128
    output_size = categories_num
    # 将 输入文本 转换成 one-hot编码
    tensor_x = lineToTensor(x)
    tensor_x.unsqueeze_(1)
    # 加载模型参数
    model = torch.load(model_param_path, weights_only=False)
    model.eval()

    # 判断是否是 LSTM（LSTM 需要 hidden 和 cell 两个状态）
    is_lstm = isinstance(model, NameClassLSTMModel)

    # 进行预测，不计算梯度
    with torch.no_grad():
        if is_lstm:
            output, hidden, cell = model(tensor_x, *model.init_hidden_and_cell())
        else:
            output, hidden = model(tensor_x, model.init_hidden())
        output = torch.softmax(output, dim=1)
        # 从预测结果在获取前三个最大的结果
        top_val, top_idx = torch.topk(output, k=3)

        # 打印结果
        print(f"{x} ---> ")
        for i in range(3):
            # 提取张量中的数值并转换为 Python 原生类型
            idx = top_idx[0][i].item()
            val = top_val[0][i].item() * 100

            # 打印（categories 直接跟索引，不要加 [0]）
            print(f"Top {i + 1}: {categories[idx]} - {val:.2f}%")


if __name__ == "__main__":
    # # 1. 训练三个模型
    # rnn_loss_list, rnn_acc_list = dm_train_rnn()
    # lstm_loss_list, lstm_acc_list = dm_train_lstm()
    # gru_loss_list, gru_acc_list = dm_train_gru()
    #
    # # 2. 把数据整理成字典
    # results = {
    #     "RNN": {"loss": rnn_loss_list, "acc": rnn_acc_list},
    #     "LSTM": {"loss": lstm_loss_list, "acc": lstm_acc_list},
    #     "GRU": {"loss": gru_loss_list, "acc": gru_acc_list},
    # }
    #
    # # 3. 创建文件夹并保存为 json 文件
    # os.makedirs("./model", exist_ok=True)
    # save_file = "./model/training_results.json"
    #
    # with open(save_file, "w", encoding="utf-8") as f:
    #     json.dump(results, f, indent=4)
    #
    # print(f"\n训练数据已成功保存到: {save_file}")
    #
    # # 对比图
    # compare()

    rnn = NameClassRNNModel(letters_len, 128, categories_num)
    lstm = NameClassLSTMModel(letters_len, 128, categories_num)
    gru = NameClassGRUModel(letters_len, 128, categories_num)

    predict("zhang", gru_path)
