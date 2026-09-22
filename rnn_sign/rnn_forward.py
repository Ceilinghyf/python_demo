import numpy as np
from softmax import softmax
from rnn_cell_forward import rnn_cell_forward

def rnn_forward(x, s0, parameters):

    """
    RNN 的向前传播
    参数：
    x：输入数据，维度为 (m_x, m, T)，m_x 是输入特征的数量，m 是样本数量，T 是时间步的数量
    s0：初始隐藏状态，维度为 (n_s, m)，n_s 是隐藏状态的数量，m 是样本数量
    parameters：包含权重和偏置的字典，包含以下键：
        W：与隐藏状态相乘的权重矩阵，维度为 (n_s, n_s)
        U：与输入相乘的权重矩阵，维度为 (n_s, m_x)
        V：将隐藏状态与输出相关的权重矩阵，维度为 (m_x, n_s)
        ba：隐藏状态的偏置，维度为 (n_s, 1)
        by：输出的偏置，维度为 (m_x, 1)
    """
    # 初始化缓存
    caches = []

    #根据X输入的形状，确定cell的个数[m_x,m,T]，T就是cell的个数
    #m_x是词的个数，n_s是自定义数字 
    m_x, m, T = x.shape # [3,1,4]
    #根据输出
    m_x, n_s = parameters['V'].shape # [3,5]

    #初始化s和y的输出
    s = np.zeros((n_s, m, T)) # [5,1,4]
    y = np.zeros((m_x, m, T)) # [3,1,4]

    #初始化s的第一个值
    s_next = s0 

    #根据cell的数量循环遍历，并保存每个时刻的输出
    for t in range(T):
        #调用rnn_cell_forward函数，计算每个时刻的输出
        s_next, out_pred, cache = rnn_cell_forward(x[:, :, t], s_next, parameters)
        #保存每个时刻的输出
        s[:, :, t] = s_next
        y[:, :, t] = out_pred
        caches.append(cache)

    return s, y, caches

if __name__ == "__main__":
    np.random.seed(1)
    x = np.random.randn(3, 1, 4)  # 输入数据 (m_x=3, m=1, T=4)
    s0 = np.random.randn(5, 1)  # 初始隐藏状态 (n_s=5, m=1)
    parameters = {
        'W': np.random.randn(5, 5),  # 权重矩阵 W (n_s=5, n_s=5)
        'U': np.random.randn(5, 3),  # 权重矩阵 U (n_s=5, m_x=3)
        'V': np.random.randn(3, 5),  # 权重矩阵 V (m_x=3, n_s=5)
        'ba': np.random.randn(5, 1),  # 偏置 ba (n_s=5, 1)
        'by': np.random.randn(3, 1)   # 偏置 by (m_x=3, 1)
    }

    s, y, caches = rnn_forward(x, s0, parameters)

    print("s =", s) 
    print("s_shape =", s.shape)
    print("y =", y)
    print("y_shape =", y.shape)
   # print("caches =", caches)