import torchvision
import torch
import PIL
import os

from matplotlib import pyplot as plt
from userFuncs import img2correct_format#导入自用函数
from model import net#引入模型
from PIL import Image


#===========参数初始化=============#
relativePth = "../imgs/white_background/"
imgspth = [relativePth+i for i in os.listdir(os.path.abspath(relativePth))]#读入图片名称并加上相对路径，放入列表


#===========网络模型导入============#
netTest = net()
netTest.load_state_dict(torch.load("../historyModel/MNIST_49th_epoch_model_lr0.001.pth"))
# print(netTest)#测试导入结果


#=======实际应用测试图片测试========#
img_outcomes = []#存储计算结果们
targets = [i[0] for i in os.listdir(os.path.abspath(relativePth))]#根据文件名获取正确的label

for imgpth in imgspth:
    img_tensor = img2correct_format(imgpth)#转为合适格式
    netTest.eval()#测试开始
    with torch.no_grad():# 取消梯度更新节省计算资源
        outputs = netTest(img_tensor)# 通过网络

    img_outcomes.append(outputs.argmax(1).item())#得到神经网络求取结果

print(img_outcomes)#展示结果

#==========结果绘图可视化=============#
plt.figure("测试结果",figsize=(24,16))
for i in range(len(img_outcomes)):
    plt.subplot(2,7,i+1)
    imgtmp = Image.open(imgspth[i])#依次打开图片
    plt.imshow(imgtmp)#展示图片
    plt.axis("off")#关闭坐标系
    plt.title("Test outcome:{}".format(img_outcomes[i]))
plt.savefig('../imgs/outcome/')#保存
plt.show()

