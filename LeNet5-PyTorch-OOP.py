# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ==============================================================================
# 【重构思路概述】
# 1. 原代码的问题：原来的 train_model 函数太“臃肿”了，里面塞了设备判断、权重初始化、
#    模型训练、准确率验证，而且外面还散落着数据加载和画图的代码。
# 2. 面向对象封装：把“训练”这件事抽象成一个类 LeNet5Trainer。
#    - 数据、网络、超参数全部作为类的属性（self.xxx）存起来，随用随取。
#    - 把原来一个大循环，拆成训练（train_epoch）和验证（validate）两个独立的方法。
# 3. 职责分离：数据加载和网络构建提取成独立函数，Trainer 类只专注训练，
#    这样代码看着不累，也符合“高内聚、低耦合”的思想。
# ==============================================================================

# ====================== 0. 超参数与全局配置 =========================
BATCH_SIZE = 256
LR = 0.1
NUM_EPOCHS = 5
# 【优化思路】把原来写死在代码里的魔法数字提出来作为全局常量，方便以后统一修改
MEAN = 0.2860
STD = 0.3530

# ====================== 1. 数据加载模块 =========================
def load_data_fashion_mnist(batch_size, resize=None):
    """独立的函数，只负责下数据"""
    trans = [transforms.ToTensor(), transforms.Normalize((MEAN,), (STD,))]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    
    train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=trans)
    test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=trans)
    
    train_iter = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_iter = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_iter, test_iter

# ====================== 2. 网络结构模块 =========================
def get_lenet5_net():
    """独立的函数，只负责拼网络"""
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

# ====================== 3. 核心重构：面向对象的训练类 ========================
class LeNet5Trainer:
    """
    将原本的 train_model 函数重构为类。
    以前函数是“用完即走”，现在类可以保存训练过程中的状态（比如 history）。
    """
    def __init__(self, net, train_iter, test_iter, num_epochs=10, lr=0.1, device=None):
        # 【重构思路】将外部传入的变量转为类的属性，这样类里面的其他方法都能直接用，不用再一层层传参了。
        self.net = net
        self.train_iter = train_iter
        self.test_iter = test_iter
        self.num_epochs = num_epochs
        
        # 自动选择设备
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)
        
        # 【重构思路】把原函数里的初始化逻辑（权重初始化、损失、优化器）直接放在 __init__ 里，实例化的时候就准备好。
        self._init_weights()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.net.parameters(), lr=lr, momentum=0.9)
        
        # 【重构思路】不再需要 return history，它现在是类的内部状态，画图的时候自己拿自己就行了。
        self.history = {'train_loss': [], 'train_acc': [], 'test_acc': []}

    def _init_weights(self):
        """【重构思路】加个下划线表示这是“私有方法”，只给类内部用。"""
        def init_func(m):
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(m.weight)
        self.net.apply(init_func)

    def train_epoch(self):
        """【重构思路】把原来混在一起的训练循环单独抽出来，只算当前的 loss 和 acc。"""
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
        """【重构思路】把测试/验证的逻辑也抽出来，保持职责单一。"""
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
        """【重构思路】主流程只负责调度：训练一轮 -> 验证一轮 -> 存数据 -> 打印。代码非常清晰。"""
        print(f'开始训练，使用设备: {self.device}')
        for epoch in range(self.num_epochs):
            # 直接调用内部的训练和验证方法
            train_loss, train_acc = self.train_epoch()
            test_acc = self.validate()
            
            # 统一保存到 self.history 中
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['test_acc'].append(test_acc)
            
            print(f'Epoch [{epoch+1}/{self.num_epochs}], '
                  f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
        print('训练完成！')

    def plot_history(self):
        """【重构思路】原来画图要传一堆参数，现在直接在类内部拿 self.history 画图，特别爽。"""
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

    def show_predictions(self, num_images=8):
        """【重构思路】把预测可视化也封装成方法，外部调用只需要一句 trainer.show_predictions()。"""
        self.net.eval()
        images, labels = next(iter(self.test_iter))
        images, labels = images[:num_images].to(self.device), labels[:num_images]
        
        with torch.no_grad():
            outputs = self.net(images)
            _, predicted = torch.max(outputs, 1)
            
        plt.figure(figsize=(12, 3))
        for idx in range(num_images):
            plt.subplot(1, num_images, idx+1)
            img_show = images[idx].cpu().squeeze() * STD + MEAN
            plt.imshow(img_show, cmap='gray')
            plt.title(f"True:{labels[idx]}\nPred:{predicted[idx].item()}")
            plt.axis('off')
        plt.tight_layout()
        plt.show()


# ====================== 4. 独立运行入口 ========================
if __name__ == '__main__':
    # 【重构思路】底下的代码就像搭积木，步骤极其清晰：
    # 建模型 -> 拿数据 -> 扔给 Trainer 训练 -> 调用画图方法

    # 1. 实例化网络并打印结构 (保留了原代码的验证逻辑)
    net = get_lenet5_net()
    X = torch.rand(size=(1, 1, 28, 28), dtype=torch.float32)
    for layer in net:
        X = layer(X)
        print(layer.__class__.__name__, 'output shape: \t', X.shape)

    # 2. 加载数据
    train_iter, test_iter = load_data_fashion_mnist(batch_size=BATCH_SIZE)

    # 3. 实例化重构后的类
    trainer = LeNet5Trainer(
        net=net, 
        train_iter=train_iter, 
        test_iter=test_iter, 
        num_epochs=NUM_EPOCHS, 
        lr=LR
    )

    # 4. 调用类的方法执行训练、画图、看预测结果
    trainer.fit()
    trainer.plot_history()
    trainer.show_predictions()