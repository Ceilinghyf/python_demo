import h5py  # 用于读取 .h5 格式的数据集文件（常用来存储大量的矩阵数据）
import numpy as np  # 进行矩阵和数组运算。
import tensorflow as tf # 深度学习框架，搭建卷积神经网络 由于VS Code上装的python版本为3.14，而TensorFlow 官方还没有发布支持 Python3.14 的版本，仅支持到Python3.12 ；
from tensorflow import keras # keras是tf的高层接口，写网络更简单
import matplotlib.pyplot as plt # 绘图库，画准确率、损失曲线，显示图片

# ===================== 1.加载数据集 =====================
def load_dataset():
    train_dataset = h5py.File('train_signs.h5', "r") #"r"：只读（read） "w"：写入（write）  "a"：追加（append）
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # 把h5文件里面图片数据读取出来，转成numpy数组
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # 把h5文件里面图片数据读取出来，转成numpy数组

    """ 
    `train_set_x`：存放全部训练图片，多维数组 `(1080,64,64,3)`
    `train_set_y`：存放训练图片对应的标签，一维数组 `(1080,)`
    `test_set_x`：测试图片 存放全部测试图片，多维数组 `(120,64,64,3)`
    `test_set_y`：测试标签 存放测试图片对应的标签，一维数组 `(120,)`
    `list_classes`：类别名称数组，这里就是 0~5 
    """

    test_dataset = h5py.File('test_signs.h5', "r") # 打开测试集文件
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # 测试图片
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # 测试标签

    classes = np.array(test_dataset["list_classes"][:]) # 读取类别信息，0,1,2,3,4,5一共6个手势

    # 调整标签维度 深度学习框架（如 TensorFlow/Keras）中，模型对输入和输出的形状有严格的数学匹配要求，所以必须进行维度调整，否则会维度不匹配报错。
    # reshape 修改数组维度
    # 原本标签是一维 (m,)，改成二维 (1, m)，m代表样本数量
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0])) # shape[0] 就是取这个数组的第 0 维的长度（即样本数量）,在这里是1080.
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0])) # reshape((1, test_set_y_orig.shape[0]))把原本并排排列的 120 个数字，强行塞进一个“1行120列”的矩阵里； 

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

# 调用函数，接收返回的5组数据
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

# 归一化像素 0~1
# 图片像素原本0‑255，除以255归一化到0~1之间
# 神经网络输入数值小，训练收敛更快，这是图像处理常规操作
X_train = X_train_orig / 255.
X_test = X_test_orig / 255.

# 转置标签，变成 (m,1)
# Y_train_orig 形状 (1, m)，.T代表矩阵转置，变成 (m,1)
# keras的 sparse 交叉熵要求标签是这种格式
Y_train = Y_train_orig.T
Y_test = Y_test_orig.T

# 打印数据集形状
# print打印输出数据的形状，用来检查数据有没有读错，调试用
print(f"训练集图片 shape: {X_train.shape}") #f-string输出格式 把里面 {} 包围的内容当成变量或表达式来执行，并把结果填进去。
print(f"测试集图片 shape: {X_test.shape}")
print(f"训练标签 shape: {Y_train.shape}")
print(f"测试标签 shape: {Y_test.shape}")
print(f"类别数量: {len(classes)}") # 计算类别长度并填入

# ===================== 2.搭建CNN模型 =====================
# Sequential：顺序模型，一层接一层堆叠网络，简单网络首选
model = tf.keras.Sequential([
    # 卷积层1 + 池化
    # 卷积层1
    # filters=8：输出8个卷积核（8个特征图）
    # kernel_size=(4,4)：卷积核大小4×4
    # strides=(1,1)：步长1，滑动窗口每次移动1像素
    # padding="same"：same填充，卷积之后输出图片大小和输入一样
    # activation="relu"：激活函数，给网络增加非线性能力，没有激活就只是矩阵乘法
    # layers：意思是“层”。神经网络就像千层饼，这个模块里包含了各种类型的层（如全连接层 Dense、池化层 MaxPooling2D 等）。
    # Conv2D：这是最核心的。Conv 是 Convolution（卷积）的缩写，2D 代表它是处理二维数据的（比如图片的高和宽）。所以它就是一个二维卷积层。
    # 如果是处理一维数据（比如一段音频或文本），就会用 Conv1D；如果是处理三维数据（比如视频或医学CT扫描），就会用 Conv3D。处理图像，默认就用 Conv2D。
    tf.keras.layers.Conv2D(filters=8, kernel_size=(4, 4), strides=(1, 1), padding="same", activation="relu",input_shape=(64, 64, 3)),

    # 第一层池化  
    # 最大池化层 MaxPooling2D
    # pool_size=(8,8)：池化窗口大小8×8
    # strides=(8,8)，窗口每次移动8像素；作用：缩小图片尺寸，降低计算量，提取关键特征
    tf.keras.layers.MaxPooling2D(pool_size=(8, 8), strides=(8, 8), padding="same"),

    # 卷积层2 + 池化 卷积层2，16个卷积核，2×2大小
    tf.keras.layers.Conv2D(filters=16, kernel_size=(2, 2), strides=(1, 1), padding="same", activation="relu"),

    # 第二层池化
    tf.keras.layers.MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding="same"),

    # 展平 + 输出层
    tf.keras.layers.Flatten(),# Flatten展平层：把二维的特征图压扁成一维向量，送给后面全连接层
    tf.keras.layers.Dense(units=6, activation="softmax") # softmax 激活函数会把这 6 个数字经过指数运算和归一化，变成 6 个 0 到 1 之间的概率值，并且所有概率加起来等于 1。
    #二分类用sigmoid激活函数，输出0~1之间的概率值；多分类用softmax激活函数，输出每个类别的概率值，所有类别概率加起来等于1。

    #第二次测试提交和推送
    #第三次测试提交和推送
])

# ===================== 3.编译模型 =====================

model.compile(optimizer="adam", # 优化器adam，自动调节学习率，更新卷积核权重参数
              loss="sparse_categorical_crossentropy", # 损失函数，多分类损失；标签是数字0‑5，不用one‑hot编码就用这个
              metrics=["accuracy"])  # 训练过程监控指标：准确率

# 打印网络概况：每一层输出shape、参数量，方便检查网络维度是否出错
model.summary()

# ===================== 4.训练模型 =====================
# fit 就是开始训练
# X_train,Y_train：训练图片和标签
# epochs=100：完整遍历全部训练数据集100次
# batch_size=64：一次拿64张图片进去算梯度、更新权重,1080 张图片 ÷ 64 = 17 个批次（所以日志里每一轮都会显示 17/17）
# validation_data=(X_test,Y_test)：每一轮跑完，拿测试集算val_acc、val_loss
history = model.fit(X_train, Y_train, epochs=100, batch_size=64, validation_data=(X_test, Y_test))

# ===================== 5.评估测试集 =====================
print("\n===== 在测试集上评估 =====")
test_loss, test_acc = model.evaluate(X_test, Y_test)
print(f"测试集损失 = {test_loss:.4f}")
print(f"测试集准确率 = {test_acc:.4f}")

# ===================== 6.绘制训练曲线 =====================
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(history.history["accuracy"], label="train acc")
plt.plot(history.history["val_accuracy"], label="val acc")
plt.title("Accuracy")
plt.xlabel("epoch")
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history["loss"], label="train loss")
plt.plot(history.history["val_loss"], label="val loss")
plt.title("Loss")
plt.xlabel("epoch")
plt.legend()
plt.show()   
#11