# -*- coding: utf-8 -*-
"""
实验2 - 任务二：LeNet-5 卷积神经网络在 MNIST 上的实现

说明：本文件是 task 二在 MNIST 数据集上的版本，沿用 LeNet5-PyTorch-OOP.py
完全相同的 OOP 训练类设计，只把数据集由 Fashion-MNIST 换成 MNIST，
用于验证课件要求的「对 MNIST 数据集进行分类的 LeNet-5 模型」。

运行：python Task2-LeNet5-MNIST.py
输出：终端日志 + results/figures 下的曲线图与预测图
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ====================== 0. 超参数与全局配置 =========================
BATCH_SIZE = 256
LR = 0.1
NUM_EPOCHS = 5
# MNIST 官方训练集的全局均值与标准差
MEAN = 0.1307
STD = 0.3081
FIG_DIR = os.path.join('results', 'figures')

# ====================== 1. 数据加载模块 =========================
def load_data_mnist(batch_size, resize=None):
    trans = [transforms.ToTensor(), transforms.Normalize((MEAN,), (STD,))]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=trans)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=trans)

    train_iter = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_iter = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_iter, test_iter

# ====================== 2. 网络结构模块 =========================
def get_lenet5_net():
    """与 LeNet5-PyTorch-OOP.py 完全一致的七层结构"""
    return nn.Sequential(
        nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.Sigmoid(),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(16*5*5, 120), nn.Sigmoid(),
        nn.Linear(120, 84), nn.Sigmoid(),
        nn.Linear(84, 10)
    )

# ====================== 3. 面向对象的训练类 ========================
class LeNet5Trainer:
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

            print(f'Epoch [{epoch+1}/{self.num_epochs}], '
                  f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
        print('训练完成！')

    def plot_history(self, save_path=None):
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
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    def show_predictions(self, num_images=8, save_path=None):
        self.net.eval()
        images, labels = next(iter(self.test_iter))
        images, labels = images[:num_images].to(self.device), labels[:num_images]

        with torch.no_grad():
            outputs = self.net(images)
            _, predicted = torch.max(outputs, 1)

        plt.figure(figsize=(12, 2.6))
        for idx in range(num_images):
            plt.subplot(1, num_images, idx+1)
            img_show = images[idx].cpu().squeeze() * STD + MEAN
            plt.imshow(img_show, cmap='gray')
            plt.title(f"True:{labels[idx]}\nPred:{predicted[idx].item()}")
            plt.axis('off')
        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    def count_parameters(self):
        return sum(p.numel() for p in self.net.parameters() if p.requires_grad)


# ====================== 4. 独立运行入口 ========================
if __name__ == '__main__':
    # 1. 实例化网络并打印每一层的输出形状
    net = get_lenet5_net()
    X = torch.rand(size=(1, 1, 28, 28), dtype=torch.float32)
    print("--- LeNet-5 网络结构 (MNIST) ---")
    for layer in net:
        X = layer(X)
        print(layer.__class__.__name__, 'output shape: \t', X.shape)

    # 2. 加载 MNIST 数据
    train_iter, test_iter = load_data_mnist(batch_size=BATCH_SIZE)

    # 3. 实例化训练类
    trainer = LeNet5Trainer(
        net=net,
        train_iter=train_iter,
        test_iter=test_iter,
        num_epochs=NUM_EPOCHS,
        lr=LR
    )
    print(f'可训练参数总量: {trainer.count_parameters():,}')

    # 4. 训练、画图、看预测
    trainer.fit()
    trainer.plot_history(os.path.join(FIG_DIR, 'Task2_LeNet5_MNIST_Training_Curves.png'))
    trainer.show_predictions(save_path=os.path.join(FIG_DIR, 'Task2_LeNet5_MNIST_Prediction_Samples.png'))

    best = max(trainer.history['test_acc'])
    print(f'最终 Test Acc: {trainer.history["test_acc"][-1]*100:.2f}%  '
          f'最好 Test Acc: {best*100:.2f}%')
