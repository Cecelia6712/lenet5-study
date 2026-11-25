# -*- coding: utf-8 -*-
"""
实验2 - 任务一：全连接网络在 Fashion-MNIST 上的实现
注：此文件复用了重构后的 ModelTrainer 类，体现了面向对象的高复用性。
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ====================== 0. 超参数与全局配置 =========================
BATCH_SIZE = 256
LR = 0.1
NUM_EPOCHS = 5
MEAN = 0.2860
STD = 0.3530

# ====================== 1. 数据加载模块 =========================
def load_data_fashion_mnist(batch_size, resize=None):
    trans = [transforms.ToTensor(), transforms.Normalize((MEAN,), (STD,))]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)

    train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=trans)
    test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=trans)

    train_iter = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_iter = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_iter, test_iter

# ====================== 2. 任务一核心：定义全连接网络 (MLP) =========================
def get_mlp_net():
    """搭建全连接神经网络"""
    return nn.Sequential(
        nn.Flatten(),                     # 将 1x28x28 展平为 784 维向量
        nn.Linear(784, 256), nn.ReLU(),   # 隐藏层 1
        nn.Linear(256, 128), nn.ReLU(),   # 隐藏层 2
        nn.Linear(128, 10)                # 输出层 (10个类别)
    )

# ====================== 3. 通用训练类 (复用了你的重构成果) ========================
class ModelTrainer:
    def __init__(self, net, train_iter, test_iter, num_epochs=10, lr=0.1, device=None):
        self.net = net
        self.train_iter = train_iter
        self.test_iter = test_iter
        self.num_epochs = num_epochs

        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)

        self._init_weights()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.net.parameters(), lr=lr, momentum=0.9)
        self.history = {'train_loss': [], 'train_acc': [], 'test_acc': []}

    def _init_weights(self):
        def init_func(m):
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(m.weight)
        self.net.apply(init_func)

    def train_epoch(self):
        self.net.train()
        loss_sum, correct, total = 0.0, 0, 0

        for X, y in self.train_iter:
            X, y = X.to(self.device), y.to(self.device)

            self.optimizer.zero_grad()
            y_hat = self.net(X)
            loss = self.criterion(y_hat, y)
            loss.backward()
            self.optimizer.step()

            loss_sum += loss.item() * X.size(0)
            _, predicted = torch.max(y_hat.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

        return loss_sum / total, correct / total

    def validate(self):
        self.net.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X, y in self.test_iter:
                X, y = X.to(self.device), y.to(self.device)
                y_hat = self.net(X)
                _, predicted = torch.max(y_hat, dim=1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        return correct / total

    def fit(self):
        print(f'开始训练，使用设备: {self.device}')
        for epoch in range(self.num_epochs):
            train_loss, train_acc = self.train_epoch()
            test_acc = self.validate()

            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['test_acc'].append(test_acc)

            print(f'Epoch [{epoch+1}/{self.num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
        print('训练完成！')

    def plot_history(self):
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(range(1, self.num_epochs+1), self.history['train_loss'], 'b-', label='Train Loss')
        plt.xlabel('Epoch'), plt.ylabel('Loss')
        plt.title('Training Loss'), plt.legend(), plt.grid(True)
        plt.subplot(1, 2, 2)
        plt.plot(range(1, self.num_epochs+1), self.history['train_acc'], 'r-', label='Train Acc')
        plt.plot(range(1, self.num_epochs+1), self.history['test_acc'], 'g-', label='Test Acc')
        plt.xlabel('Epoch'), plt.ylabel('Accuracy')
        plt.title('Accuracy'), plt.legend(), plt.grid(True)
        plt.tight_layout()
        plt.show()

# ====================== 4. 独立运行入口 ========================
if __name__ == '__main__':
    # 1. 实例化网络并打印结构
    net = get_mlp_net()
    X = torch.rand(size=(1, 1, 28, 28), dtype=torch.float32)
    print("--- 全连接网络结构 ---")
    for layer in net:
        X = layer(X)
        print(layer.__class__.__name__, 'output shape: \t', X.shape)

    # 2. 加载数据
    train_iter, test_iter = load_data_fashion_mnist(batch_size=BATCH_SIZE)

    # 3. 实例化训练类并执行
    trainer = ModelTrainer(net=net, train_iter=train_iter, test_iter=test_iter, num_epochs=NUM_EPOCHS, lr=LR)
    trainer.fit()
    trainer.plot_history()
