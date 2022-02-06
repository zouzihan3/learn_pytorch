import torch
import torch.optim

from model import *
from dataload import *
from torch.utils.tensorboard import SummaryWriter

#创建可视化
writer = SummaryWriter("../logs/logs_trian_lr0.005_epoch50")

#选择cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")#如果cuda可行即cuda
print("Your device is {}.".format(device))

#创建网络模型
Net = net()
Net.to(device)

#损失函数此分类问题我们使用交叉熵作为损失函数
lossFunc = nn.CrossEntropyLoss()
lossFunc.to(device)
#优化器
learningRate = 0.005#学习速率超参数，一般取的越小训练越慢，但又有较高的精准度
optimizer = torch.optim.SGD(Net.parameters(),lr=learningRate)

#训练网络的一些参数
#训练的次数
trainStep = 0
#测试的次数
testStep = 0
#训练的轮数
epoch = 50


for i in range(epoch):
    print("========第{}轮训练开始=========".format(i+1))

    #============训练开始=============#
    Net.train() #不是torch.train()
    for data in trainLoader:
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        # useCuda(imgs)#使用gpu
        # useCuda(targets)
        outputs = Net(imgs)
        loss = lossFunc(outputs, targets)#根据网络结果与targets目标计算损失函数

        #优化器调优
        optimizer.zero_grad()#梯度归零防止梯度累加
        loss.backward()#将参数反向传播用于优化梯度
        optimizer.step()#梯度优化

        #日志
        trainStep = trainStep + 1  # 训练次数加1
        if trainStep % 100 == 0:#每百次进行一次日志
            print("训练次数{}：Loss is {}".format(trainStep,loss.item()))
            writer.add_scalar('train_loss', loss.item(), trainStep)#在tensorboard中记录

    #==============测试开始==============#
    Net.eval()#最好加上养成规范
    testLoss = 0#初始化测试误差
    epochTestAccuracy = 0#初始化正确总个数
    with torch.no_grad():#取消梯度更新节省算力
        for data in testLoader:
            imgs, targets = data
            imgs = imgs.to(device)#切换到指定硬件
            targets = targets.to(device)
            outputs = Net(imgs)
            loss = lossFunc(outputs, targets)#计算误差函数但后无optimizer仅测试
            # 误差计算
            testLoss = testLoss + loss.item()#计算累计误差量
            tmpAccuracy = (outputs.argmax(1) == targets).sum()#计算当前数据中正确个数
            # 计算正确率的常用方法
            # 其中.argmax（1）以行的方式求结果中概率最大的索引并与targets数组对位比较，对为ture反之false,
            # 最后用.sum()取得该结果中正确个数

            epochTestAccuracy = tmpAccuracy + epochTestAccuracy#计算当前epoch下的总正确个数
    Accuracy = epochTestAccuracy / length_testData#正确率

    ## 日志
    print('该轮测试累计误差为{}'.format(testLoss))
    print(('该轮的正确率为{}'.format(Accuracy)))
    #画板中展示
    writer.add_scalar('Each Epoch Test Accuracy ', Accuracy, testStep)
    writer.add_scalar("Each Epoch Test Loss", testLoss, testStep)
    testStep = testStep + 1

    #模型存储
    torch.save(Net.state_dict(), "../historyModel/MNIST_{}th_epoch_model_lr0.001.pth".format(i))

writer.close()





