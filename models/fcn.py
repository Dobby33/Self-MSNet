import torch.nn as nn
from .common import add_module


# 0.004s taken for {fcn}
def fcn(num_input_channels=200, num_output_channels=1, num_hidden=1000):
    # Fully Connected Network, fcn
    model = nn.Sequential()
    '''
        Sequential一个有序的容器，神经网络模块将按照在传入构造器的顺序依次被添加到计算图中执行，
        同时以神经网络模块为元素的有序字典也可以作为传入参数
        model = nn.Sequential( 
                      nn.Conv2d(1,20,5),
                      nn.ReLU(),
                      nn.Conv2d(20,64,5),
                      nn.ReLU()
                )
    '''
    model.add(nn.Linear(num_input_channels, num_hidden, bias=True))
    '''
    ReLU6就是普通的ReLU但是限制最大输出值为6（对输出值做clip），这是为了在移动端设备float16的低精度的时候，
    也能有很好的数值分辨率，如果对ReLU的激活范围不加限制，输出范围为0到正无穷，如果激活值非常大，分布在一个
    很大的范围内，则低精度的float16无法很好地精确描述如此大范围的数值，带来精度损失
    '''
    model.add(nn.ReLU6())
    model.add(nn.Linear(num_hidden, num_output_channels))
    model.add(nn.Softmax())
    return model











