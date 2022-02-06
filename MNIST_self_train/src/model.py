import torch
import torch.nn as nn
# 根据网上资料构建网络


class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()
        self.model = nn.Sequential(# 逗号连接
            nn.Conv2d(in_channels=1,out_channels=32, kernel_size=3, stride=1,padding=1),# 应注意通道数填写
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(in_features=64*7*7, out_features=128),
            nn.Linear(in_features=128, out_features=10),
        )

    def forward(self, x):
        x = self.model(x)
        return x


if __name__ == "__main__":
    Net = net()
    input = torch.ones((64, 1, 28, 28))
    output = Net(input)
    print(output.shape)