# 1 加载必要的库
import torch
import torch.nn as nn  # 网络模型
import torch.nn.functional as F
import torch.optim as optim  # 优化器
from torchvision import datasets, transforms  # 视觉处理
# import numpy as np

# 2 定义超参数
BATCH_SIZE = 128  # 定义每批处理的数据
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 是否用GPU还是CPU进行训练
EPOCHS = 30  # 训练数据集十轮

# 3 构建pipeline，对图像进行处理
dataset_transforms = transforms.Compose([
    transforms.ToTensor(),  # 将图片转换成Tensor（数值型容器）
    transforms.Normalize((0.1307,), (0.3081,))  # 对数据进行正则化[均值和标准差，官网提供]（当模型出现过拟合现象[过拟合不会举一反三]时，降低模型复杂度）
])

# 4 加载数据
from torch.utils.data import DataLoader

    # 下载数据集
train_set = datasets.MNIST("data", train=True, download=True, transform=dataset_transforms)

test_set = datasets.MNIST("data", train=False, download=True, transform=dataset_transforms)

    # 加载数据
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)  # 打乱数据集

test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)

# 5 构建网络模型
class Digit(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, 5)  # 1：灰度图片的通道；10：输出通道；5：5*5卷积核
        self.conv2 = nn.Conv2d(10, 20, 3)  # 10：输入通道；20：输出通道；3：Kernel
        self.fc1 = nn.Linear(20*10*10, 500)  # 20*20*20：输入通道；500：输出通道[全连接层]
        self.fc2 = nn.Linear(500, 10)  # 500：输入通道；10：输出通道（输出10个通道）

    def forward(self, input):  # 前向传播
        input_size = input.size(0)  # batch_size (*1*28*28)
        input = self.conv1(input)  # 输入：batch_size*1*28*28， 输出：batch*10*24*24（28-5+1）
        input = F.relu(input)  # 激活函数，未出输出，保持shape不变
        input = F.max_pool2d(input, 2, 2)  # [最大]池化层[重点看某个地方]（对图片进行压缩[降采率]降采样，大幅减少神经网络的样本量） 输入：batch*10*24*24；输出：batch*10*12*12

        input = self.conv2(input)  # 输入：batch*10*12*12；输出：batch*20*10*10（12-3+1=10）
        input = F.relu(input)  # 激活，保持shape不变

        input = input.view(input_size, -1)  # 拉平[三维数组拉成一维数组]； -1：自动计算维度，20*10*10=2000

        input = self.fc1(input)  # 全连接层， 输入：batch*2000 输出：batch*500
        input = F.relu(input)  # 激活，保持shape不变

        input = self.fc2(input)  # 输入：batch*500[上一层的输出] 输出batch*10

        output = F.log_softmax(input, dim=1)  # 损失函数，计算分类后，每个数字的概率值（求log，dim维度=1[按行来计算]）

        return output

# 6 定义优化器
model = Digit().to(DEVICE)  # 创建模型，部署到设备上

optimizer = optim.Adam(model.parameters())  # 调用Adam优化器，更新我们的模型参数，使得最终训练和测试的结果达到最优值

# 7 定义训练方法
def train_model(model, device, train_loader, optimizer, epoch):  # 传入模型，设备，数据，优化器，循环几次
    # 模型训练
    model.train()
    for batch_index, (data, target) in enumerate(train_loader):
        # 部署到DEVICE上面去
        data, target = data.to(device), target.to(device)
        # 梯度初始化为0
        optimizer.zero_grad()
        # 预测，训练后的结果
        output = model(data)
        # 计算损失
        loss = F.cross_entropy(output, target)  # 交叉熵损失，针对于多分类的任务
        # # 找到预测值（概率值最大的下标）
        # pred = output.max(1, keepdim=True)  # 1：（维度）横轴，找到每一行最大值的下标 pred = output.argmax(dim=1)
        # 反向传播
        loss.backward()
        # 参数优化
        optimizer.step()
        if batch_index % 3000 == 0:
            print("Train Epoch : {} \t Loss : {:.6f}".format(epoch, loss.item()))

# 8 定义测试方法
def test_modle(modle, device, test_loader):
    # 模型验证
    modle.eval()
    # 正确率
    correct = 0.0
    # 测试损失
    test_loss = 0.0
    # 测试
    with torch.no_grad():  # 不会计算梯度和反向传播
        for data, target in test_loader:
            # 部署到device
            data, target = data.to(device), target.to(device)
            # 测试数据
            output = model(data)
            # 计算测试损失
            test_loss += F.cross_entropy(output, target).item()
            # 找到概率值最高的下标
            pred = output.max(1, keepdim=True)[1]  # 0值，1索引
            # pred = torch.max(output, dim=1)
            # pred = output.argmax(dim=1)
            # 累计正确率
            correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(test_loader.dataset)
        print("Test —— Average loss : {:.4f}, Accuracy : {:.3f}\n".format(
            test_loss, 100.0 * correct / len(test_loader.dataset)))

# 9 调用方法7、8 输出结果
if __name__ == '__main__':
    for epoch in range(1, EPOCHS + 1):
        train_model(model, DEVICE, train_loader, optimizer, epoch)
        test_modle(model, DEVICE, test_loader)

    # 保存模型参数和结构
    torch.save(model, 'mnist_self_nor.pth')
    # # 读取
    # the_model = TheModelClass(*args, **kwargs)
    # the_model.load_state_dict(torch.load(PATH))
