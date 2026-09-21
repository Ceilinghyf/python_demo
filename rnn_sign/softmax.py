import numpy as np

def softmax(x):
    """
    计算softmax函数
    参数：
    x：输入数据，维度为 (n_y, m)，n_y 是输出类别的数量，m 是样本数量
    返回：
    s：softmax函数的输出，维度为 (n_y, m)
    """
    # 为了数值稳定性，减去每列的最大值
    x_max = np.max(x, axis=0, keepdims=True)
    e_x = np.exp(x - x_max)
    s = e_x / np.sum(e_x, axis=0, keepdims=True)
    
    return s
