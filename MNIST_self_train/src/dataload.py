import torchvision
from torch.utils.data import DataLoader
import userFuncs as fs
#数据集路径
datasetsPth = '../data'

#训练数据集
trainData = torchvision.datasets.MNIST(root=datasetsPth,#设置数据集路径
                                      train=True,#设置为训练数据集
                                      transform=torchvision.transforms.ToTensor(),#化为tensor
                                      download=True)#启动下载
#测试数据集

testData = torchvision.datasets.MNIST(root=datasetsPth,
                                      train=False,#设置为测试数据集
                                      transform=torchvision.transforms.ToTensor(),
                                      download=True)
#长度
length_trainData = len(trainData)
length_testData = len(testData)

print("Length of train datasets is {};\n"
      "Length of test datasets is {};".format(length_trainData, length_testData))

#DataLoader加载数据集
trainLoader = DataLoader(trainData, batch_size=64)
testLoader = DataLoader(testData, batch_size=64)


