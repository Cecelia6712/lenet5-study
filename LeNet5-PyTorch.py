# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


# ====================== 1. 超参数配置 =========================
BATCH_SIZE = 256    # 批次大小
LR         = 0.1    # 学习率
NUM_EPOCHS =  5     # 训练轮数


# ====================== 2. 定义LeNet-5网络结构 ================
net = nn.Sequential(
    # C1卷积层：输入1通道，输出6通道，5x5卷积核，无padding → 输出6@28x28
    nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.Sigmoid(),
    # S2池化层：2x2平均池化，步长2 → 输出6@14x14
    nn.AvgPool2d(kernel_size=2, stride=2),
    # C3卷积层：输入6通道，输出16通道，5x5卷积核，无padding → 输出16@10x10
    nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),
    # S4池化层：2x2平均池化，步长2 → 输出16@5x5
    nn.AvgPool2d(kernel_size=2, stride=2),
    # 展平：将16x5x5的特征图拉平为400维向量
    nn.Flatten(),
    # C5全连接层：400 -> 120
    nn.Linear(16*5*5, 120), nn.Sigmoid(),
    # F6全连接层：120 -> 84
    nn.Linear(120, 84), nn.Sigmoid(),
    # 输出层：84 -> 10（对应10个数字类别）
    nn.Linear(84, 10)
)

X = torch.rand(size=(1, 1, 28, 28), dtype=torch.float32)
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__,'output shape: \t',X.shape)


# ====================== 3. 数据加载与预处理 ====================
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def load_data_fashion_mnist(batch_size, resize=None):
    '''
    加载 FashionMNIST 数据集（自动下载）.
    类别：['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    '''
    # 预处理设置
    trans = [transforms.ToTensor(),                         # 转为Tensor，像素值归一化到[0,1]
             transforms.Normalize((0.2860,), (0.3530,))]    # 数据集的均值和标准差，加速收敛
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    # 加载FashionMNIST数据集（自动下载）
    train_dataset = datasets.FashionMNIST(root='./data', train=True,  download=True, transform=trans)
    test_dataset  = datasets.FashionMNIST(root='./data', train=False, download=True, transform=trans)
    # 创建数据加载器
    train_iter = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=0) # num_workers=0兼容Windows
    test_iter  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=0)
    return (train_iter, test_iter)

train_iter, test_iter = load_data_fashion_mnist(batch_size=BATCH_SIZE)


# ====================== 4. 训练分类模型 ========================
import torch.optim as optim
import matplotlib.pyplot as plt

def train_model(net, train_iter, test_iter, num_epochs=10, lr=0.1, device=None):
    '''
    训练函数
    net,                  # 要训练的网络模型
    train_iter,           # 训练集DataLoader
    test_iter,            # 测试集DataLoader
    num_epochs=10,        # 训练轮数
    lr=0.1,               # 学习率
    device=None,          # 训练设备，None则自动选择GPU/CPU
    '''
    # ---- 设备选择 ------------------
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ---- 权重初始化 ----------------
    def init_weights(m):
        if isinstance(m, (nn.Linear, nn.Conv2d)): # 检查层是否为全连接层或卷积层
            nn.init.xavier_uniform_(m.weight) # Xavier初始化，适配Sigmoid激活
    net.apply(init_weights) # 对网络的所有层应用init_weights制定的权重初始化
    net = net.to(device) # 模型移到GPU/CPU
    
    # ---- 定义损失函数与优化器 -------
    criterion = nn.CrossEntropyLoss() # 交叉熵损失，内部自动包含Softmax，无需在网络最后加
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9) # 带动量的SGD加速收敛

    # ---- 训练循环 ------------------
    # 记录训练过程数据用于绘图
    history = {'train_loss': [], 'train_acc': [], 'test_acc': []}
    
    # 评估函数
    def evaluate_accuracy(net, data_iter):
        net.eval() # 切换为评估模式，关闭Dropout/BN的训练行为
        correct, total = 0, 0
        with torch.no_grad(): # 评估时不需要计算梯度，节省内存
            for X, y in data_iter:
                X, y = X.to(device), y.to(device)
                y_hat = net(X) # y_hat是一个（batch_size x 10）形状的张量，对应10个数字类别概率
                max_score, predicted = torch.max(y_hat, dim=1) # 取概率最大的类别作为预测结果
                total += y.size(0) # 累计总样本数
                correct += (predicted == y).sum().item() # 累计正确预测的样本数，转换为Python标量
        net.train() # 切回训练模式
        return correct / total

    print(f'开始训练，使用设备: {device}')
    for epoch in range(num_epochs):
        net.train()
        train_loss_sum, train_correct, train_total = 0.0, 0, 0
        
        for i, (X, y) in enumerate(train_iter):
            X, y = X.to(device), y.to(device)
            # 前向传播
            optimizer.zero_grad() # 梯度清零
            y_hat = net(X)
            loss = criterion(y_hat, y)
            # 反向传播与参数更新
            loss.backward()
            optimizer.step()
            # 累计训练指标
            train_loss_sum += loss.item() * X.size(0)
            _, predicted = torch.max(y_hat.data, 1)
            train_total += y.size(0)
            train_correct += (predicted == y).sum().item()
        
        # 计算本轮平均指标
        train_loss = train_loss_sum / train_total
        train_acc = train_correct / train_total
        test_acc = evaluate_accuracy(net, test_iter)
        # 记录训练历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        # 打印日志
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
    print('训练完成！')
    return net, history

trained_net, history = train_model(net, train_iter, test_iter, NUM_EPOCHS, LR)


# ====================== 5. 可视化训练过程 =====================
def plot_history(history, num_epochs=10):
    plt.figure(figsize=(12, 4))
    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs+1), history['train_loss'], 'b-', label='Train Loss')
    plt.xlabel('Epoch'), plt.ylabel('Loss')
    plt.title('Training Loss'), plt.legend(), plt.grid(True)
    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs+1), history['train_acc'], 'r-', label='Train Acc')
    plt.plot(range(1, num_epochs+1), history['test_acc'], 'g-', label='Test Acc')
    plt.xlabel('Epoch'), plt.ylabel('Accuracy')
    plt.title('Accuracy'), plt.legend(), plt.grid(True)
    
    plt.tight_layout()
    plt.show()

plot_history(history, NUM_EPOCHS)


# ====================== 6. 可视化预测结果 ======================
def show_predicted(net, test_iter):
    def imshow(img, title):
        img = img * 0.3530 + 0.2860 # 反归一化
        plt.imshow(img.cpu().squeeze(), cmap='gray')
        plt.title(title)
        plt.axis('off')
    
    device = next(net.parameters()).device
    # 取一批测试集数据
    test_images, test_labels = next(iter(test_iter))
    test_images, test_labels = test_images[:8].to(device), test_labels[:8]
    outputs = net(test_images)
    _, predicted = torch.max(outputs, 1)
    
    # 显示前8张预测结果
    plt.figure(figsize=(12, 3))
    for idx in range(8):
        plt.subplot(1, 8, idx+1)
        title = f"True:{test_labels[idx]}\nPred:{predicted[idx].item()}"
        imshow(test_images[idx].cpu(), title)
    plt.tight_layout()
    plt.show()

show_predicted(trained_net, test_iter)

