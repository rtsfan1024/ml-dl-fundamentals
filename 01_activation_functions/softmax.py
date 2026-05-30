# encoding=utf-8

import numpy as np

def softmax(list):
    """
    计算机处理“数”时，数值必须在4字节或8字节的有限数据宽度内，因此，超大值无法表示（称为溢出）
    为了防止溢出nan(not a number)，一般会减去输入信号中的最大值
    """
    max = np.max(list) # 信号（列表）中的最大值
    e_k = np.exp(list - max) # 溢出对策，指数函数list-信号（列表）中的最大值max
    sum_e_i = np.sum(e_k) # 指数函数的和
    y = e_k / sum_e_i
    return y

print(softmax([0.3, 2.9, 4.0])) # softmax([0.3,2.9,4.0])
print([float("{:.2f}".format(i)) for i in softmax([0.3,2.9,4.0])]) # softmax([0.3,2.9,4.0])



