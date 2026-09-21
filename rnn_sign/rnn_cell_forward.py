import numpy as np 
from softmax import softmax

def rnn_cell_forward(x_t, s_prev, parameters):
    """
    单个RNN—cell 的向前传播 
    参数：x_t：当前时间步的输入数据，维度为 (n_x, m)，n_x 是输入特征的数量，m 是样本数量
    参数：s_prev：前一时间步的隐藏状态，维度为 (n_a, m)，n_a 是隐藏状态的数量，m 是样本数量
    参数：parameters：包含权重和偏置的字典，包含以下键：
    参数：W：与隐藏状态相乘的权重矩阵，维度为 (n_a, n_a)
    参数：U：与输入相乘的权重矩阵，维度为 (n_a, n_x)
    参数：V：将隐藏状态与输出相关的权重矩阵，维度为 (n_y, n_a)
    参数：ba：隐藏状态的偏置，维度为 (n_a, 1)
    参数：by：输出的偏置，维度为 (n_y, 1)
    """
    # 获取参数
    U = parameters['U']
    W = parameters['W']
    V = parameters['V']
    ba = parameters['ba']
    by = parameters['by']

    # 计算激活函数
    s_next = np.tanh(np.dot(W, s_prev) + np.dot(U, x_t) + ba)

    # 计算当前的cell输出预测结果
    out_pred = softmax(np.dot(V, s_next) + by)

    # 记录每一层的值，用于反向传播计算
    cache = (s_next, s_prev, x_t, U, W, V, ba, by)

    return s_next, out_pred, cache

if __name__ == "__main__":
    # 测试 rnn_cell_forward 函数
    np.random.seed(1)
    x_t = np.random.randn(3, 1)  # 输入数据 (n_x=3, m=1)
    s_prev = np.random.randn(3, 1)  # 前一时间步的隐藏状态 (n_a=3, m=1)
    parameters = {
        'W': np.random.randn(3, 3),  # 权重矩阵 W (n_a=3, n_a=3)
        'U': np.random.randn(3, 3),  # 权重矩阵 U (n_a=3, n_x=3)
        'V': np.random.randn(3, 3),  # 权重矩阵 V (n_y=3, n_a=3)
        'ba': np.random.randn(3, 1),  # 偏置 ba (n_a=3, 1)
        'by': np.random.randn(3, 1)   # 偏置 by (n_y=3, 1)
    }

    s_next, out_pred, cache = rnn_cell_forward(x_t, s_prev, parameters)

    print("s_next =", s_next) 
    print("s_next_shape =", s_next.shape)
    print("out_pred =", out_pred)
    print("out_pred_shape =", out_pred.shape)

    #此套代码仅是用来验证矩阵维度、公式有没有写错