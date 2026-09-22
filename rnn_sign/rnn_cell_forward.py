import numpy as np 
from softmax import softmax

def rnn_cell_forward(x_t, s_prev, parameters):
    """
    单个RNN—cell 的向前传播 
    参数：x_t：当前时间步的输入数据，维度为 (m_x, m)，m_x 是输入特征的数量，m 是样本数量
    参数：s_prev：前一时间步的隐藏状态，维度为 (n_s, m)，n_s 是隐藏状态的数量，m 是样本数量
    参数：parameters：包含权重和偏置的字典，包含以下键：
    参数：W：与隐藏状态相乘的权重矩阵，维度为 (n_s, n_s)
    参数：U：与输入相乘的权重矩阵，维度为 (n_s, m_x)
    参数：V：将隐藏状态与输出相关的权重矩阵，维度为 (m_x, n_s)
    参数：ba：隐藏状态的偏置，维度为 (n_s, 1)
    参数：by：输出的偏置，维度为 (m_x, 1)
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

    # 如果别的文件 import 导入这个文件，这段代码不会跑。
    # 测试 rnn_cell_forward 函数
    # 缩进很重要！if下面所有缩进的代码，才是受这个条件控制的代码；
    # __name__、__main__前后都是两个下划线，一个都不能少，不能改名字
if __name__ == "__main__":
    # 固定随机种子，每次运行随机数都一样，方便调试复现。
    np.random.seed(1)
    x_t = np.random.randn(3, 1)  # 输入数据 (n_x=3, m=1)
    s_prev = np.random.randn(5, 1)  # 前一时间步的隐藏状态 (n_s=5, m=1)
    parameters = {
        'W': np.random.randn(5, 5),  # 权重矩阵 W (n_s=5, n_s=5)
        'U': np.random.randn(5, 3),  # 权重矩阵 U (n_s=5, m_x=3)
        'V': np.random.randn(3, 5),  # 权重矩阵 V (m_x=3, n_s=5)
        'ba': np.random.randn(5, 1),  # 偏置 ba (n_s=5, 1)
        'by': np.random.randn(3, 1)   # 偏置 by (m_x=3, 1)
    }

    s_next, out_pred, cache = rnn_cell_forward(x_t, s_prev, parameters)

    print("s_next =", s_next) 
    print("s_next_shape =", s_next.shape)
    print("out_pred =", out_pred)
    print("out_pred_shape =", out_pred.shape)

    #此套代码仅是用来验证矩阵维度、公式有没有写错