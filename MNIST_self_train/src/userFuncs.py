'''此为我自主写的一些常用小功能'''
import PIL
import torch
import torchvision
from PIL import Image


#将导入的图片路径设置为适合于网络的输入的形式
def img2correct_format(imgpth):
    img = Image.open(imgpth)  # 以pil打开图片方便后续变形
    img = PIL.ImageOps.invert(img)  # 白底黑字能够取反色
    img = img.convert("L")  # 灰度化以单通道导入以适应MNIST的特性
    # print(img)#验证格式是否正确
    # img.show()#展示图片

    # 更改图规格
    transform = torchvision.transforms.Compose([torchvision.transforms.Resize((28, 28)),
                                                torchvision.transforms.ToTensor()])
    img = transform(img)
    # print(img.shape)#验证规格

    img = torch.reshape(img, (1, 1, 28, 28))  # 转变为需要类型的tensor，直接三通道转不行的，需要先灰度导入

    return img

