import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os

# Your existing model classes (with minor modifications for MNIST)
class DenseLayer(nn.Module):
    """DenseNet的基本密集层 - MNIST优化版本"""
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(DenseLayer, self).__init__()
        # 第一个1x1卷积（瓶颈层）
        self.norm1 = nn.BatchNorm2d(num_input_features)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(num_input_features, bn_size * growth_rate,
                              kernel_size=1, stride=1, bias=False)
        
        # 第二个3x3卷积
        self.norm2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate,
                              kernel_size=3, stride=1, padding=1, bias=False)
        
        self.drop_rate = drop_rate

    def forward(self, x):
        # 第一层：BN -> ReLU -> Conv1x1
        out = self.norm1(x)
        out = self.relu1(out)
        out = self.conv1(out)
        
        # 第二层：BN -> ReLU -> Conv3x3
        out = self.norm2(out)
        out = self.relu2(out)
        out = self.conv2(out)
        
        # 应用Dropout
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        
        # 密集连接：将输入和输出在通道维度拼接
        return torch.cat([x, out], 1)

class DenseBlock(nn.Module):
    """DenseNet的密集块"""
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        
        for i in range(num_layers):
            layer = DenseLayer(
                num_input_features + i * growth_rate, 
                growth_rate, 
                bn_size, 
                drop_rate
            )
            self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class Transition(nn.Module):
    """DenseNet的过渡层"""
    def __init__(self, num_input_features, num_output_features):
        super(Transition, self).__init__()
        self.norm = nn.BatchNorm2d(num_input_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(num_input_features, num_output_features,
                             kernel_size=1, stride=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        out = self.norm(x)
        out = self.relu(out)
        out = self.conv(out)
        out = self.pool(out)
        return out

class InceptionBlock(nn.Module):
    """Inception模块 - MNIST优化版本"""
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(InceptionBlock, self).__init__()
        
        # 1x1卷积分支
        self.branch1 = nn.Conv2d(in_channels, ch1x1, kernel_size=1)
        
        # 1x1 -> 3x3卷积分支
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, ch3x3red, kernel_size=1),
            nn.BatchNorm2d(ch3x3red),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch3x3red, ch3x3, kernel_size=3, padding=1)
        )
        
        # 1x1 -> 3x3卷积分支（替代5x5）
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, ch5x5red, kernel_size=1),
            nn.BatchNorm2d(ch5x5red),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch5x5red, ch5x5, kernel_size=3, padding=1)
        )
        
        # 3x3最大池化 -> 1x1卷积分支
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1)
        )
        
        self.bn = nn.BatchNorm2d(ch1x1 + ch3x3 + ch5x5 + pool_proj)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        
        outputs = torch.cat([branch1, branch2, branch3, branch4], 1)
        outputs = self.bn(outputs)
        outputs = self.relu(outputs)
        
        return outputs

class DenseInceptionNet_MNIST(nn.Module):
    """DenseNet + Inception 混合网络（MNIST优化版本）"""
    def __init__(self, growth_rate=8, block_config=(4, 6, 8, 6),
                 num_init_features=16, bn_size=4, drop_rate=0.1, num_classes=10):
        super(DenseInceptionNet_MNIST, self).__init__()
        
        # 初始卷积层（适配28x28输入，单通道）
        self.conv0 = nn.Conv2d(1, num_init_features, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm0 = nn.BatchNorm2d(num_init_features)
        self.relu0 = nn.ReLU(inplace=True)
        
        # 第一个Inception模块（减小通道数）
        self.inception1 = InceptionBlock(num_init_features, 4, 4, 8, 2, 4, 4)
        num_features = 4 + 8 + 4 + 4  # 20
        
        # 构建DenseNet块
        self.dense_blocks = nn.ModuleList()
        self.transitions = nn.ModuleList()
        self.inception_blocks = nn.ModuleList()
        
        for i, num_layers in enumerate(block_config):
            # DenseBlock
            block = DenseBlock(
                num_layers=num_layers, 
                num_input_features=num_features,
                bn_size=bn_size, 
                growth_rate=growth_rate, 
                drop_rate=drop_rate
            )
            self.dense_blocks.append(block)
            num_features = num_features + num_layers * growth_rate
            
            if i != len(block_config) - 1:
                # Transition层
                trans = Transition(
                    num_input_features=num_features,
                    num_output_features=num_features // 2
                )
                self.transitions.append(trans)
                num_features = num_features // 2
                
                # Inception模块（逐渐增大通道数）
                if i == 0:
                    inception = InceptionBlock(num_features, 8, 8, 16, 4, 8, 8)
                    inception_out = 8 + 16 + 8 + 8  # 40
                elif i == 1:
                    inception = InceptionBlock(num_features, 16, 16, 32, 8, 16, 16)
                    inception_out = 16 + 32 + 16 + 16  # 80
                else:
                    inception = InceptionBlock(num_features, 32, 32, 64, 16, 32, 32)
                    inception_out = 32 + 64 + 32 + 32  # 160
                
                self.inception_blocks.append(inception)
                num_features = inception_out
        
        # 最终批归一化
        self.norm_final = nn.BatchNorm2d(num_features)
        
        # 分类器
        self.classifier = nn.Linear(num_features, num_classes)
        
        # 权重初始化
        self._initialize_weights()
    
    def _initialize_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # 初始卷积
        features = self.conv0(x)
        features = self.norm0(features)
        features = self.relu0(features)
        
        # 第一个Inception
        features = self.inception1(features)
        
        # DenseNet块序列
        for i, dense_block in enumerate(self.dense_blocks):
            features = dense_block(features)
            
            if i < len(self.transitions):
                features = self.transitions[i](features)
                features = self.inception_blocks[i](features)
        
        # 最终处理
        out = self.norm_final(features)
        out = F.relu(out, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        
        return out

# 训练和评估函数
def train_epoch(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
        
        pbar.set_postfix({
            'Loss': f'{train_loss/(batch_idx+1):.4f}',
            'Acc': f'{100.*correct/total:.2f}%'
        })
    
    return train_loss / len(train_loader), 100. * correct / total

def test_epoch(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    
    test_loss /= len(test_loader)
    accuracy = 100. * correct / total
    
    return test_loss, accuracy

def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # MNIST数据集的标准化参数
    # MNIST数据集的均值和标准差
    mnist_mean = 0.1307
    mnist_std = 0.3081
    
    # 数据预处理和增强
    train_transform = transforms.Compose([
        transforms.RandomRotation(10),  # 随机旋转
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # 随机平移
        transforms.ToTensor(),
        transforms.Normalize((mnist_mean,), (mnist_std,))
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((mnist_mean,), (mnist_std,))
    ])
    
    # 加载数据集
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=train_transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=test_transform)
    
    # 创建数据加载器
    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # 创建模型
    model = DenseInceptionNet_MNIST(
        growth_rate=8,
        block_config=(4, 6, 8, 6),
        num_init_features=16,
        bn_size=4,
        drop_rate=0.1,
        num_classes=10
    ).to(device)
    
    # 打印模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total parameters: {total_params:,}')
    print(f'Trainable parameters: {trainable_params:,}')
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)
    
    # 训练循环
    num_epochs = 25
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []
    
    best_accuracy = 0
    
    for epoch in range(1, num_epochs + 1):
        # 训练
        train_loss, train_acc = train_epoch(model, device, train_loader, optimizer, criterion, epoch)
        
        # 测试
        test_loss, test_acc = test_epoch(model, device, test_loader, criterion)
        
        # 更新学习率
        scheduler.step()
        
        # 记录结果
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)
        
        print(f'Epoch {epoch}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
        print(f'Learning Rate: {scheduler.get_last_lr()[0]:.6f}')
        print('-' * 50)
        
        # 保存最佳模型
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'New best accuracy: {best_accuracy:.2f}%')
    
    print(f'Training completed! Best accuracy: {best_accuracy:.2f}%')
    
    # 绘制训练曲线
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Test Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    # 显示一些测试样本和预测结果
    model.eval()
    with torch.no_grad():
        data, target = next(iter(test_loader))
        data, target = data.to(device), target.to(device)
        output = model(data[:9])
        pred = output.argmax(dim=1)
        
        for i in range(9):
            plt.subplot(3, 3, i+1)
            img = data[i].cpu().squeeze()
            plt.imshow(img, cmap='gray')
            plt.title(f'True: {target[i].item()}, Pred: {pred[i].item()}')
            plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('training_results.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    main()