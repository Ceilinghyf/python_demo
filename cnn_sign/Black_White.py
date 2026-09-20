import matplotlib.pyplot as plt
import numpy as np

# 手写卷积函数
def conv2d(image, kernel):
    img_h, img_w = image.shape
    k_h, k_w = kernel.shape
    # 计算输出尺寸
    out_h = img_h - k_h + 1
    out_w = img_w - k_w + 1
    output = np.zeros((out_h, out_w), dtype=np.float32) #深度学习框架（Paddle/Torch）里卷积核默认都是 float32
    
    # 滑动窗口遍历
    for i in range(out_h):
        for j in range(out_w):
            patch = image[i:i+k_h, j:j+k_w]
            output[i,j] = np.sum(patch * kernel)
    return output

#创建 50×50 矩阵，全部填充 1，图像白色
#所有行，第 30 列往后全部赋值 0，0 代表黑色 
img = np.ones([50,50], dtype='float32') #深度学习框架（Paddle/Torch）里卷积核默认都是 float32
img[:, 30:] = 0.

# 卷积核 [1, 0, -1] 竖直边缘检测，1行3列
#窗口左边亮 (1)、中间不变 (0)、右边暗 (-1)，只有遇到左右灰度突变的竖线，求和才不为 0；
# 均匀白色 / 均匀黑色区域，求和结果 = 0，变黑。
kernel = np.array([[1, 0, -1]], dtype='float32') #深度学习框架（Paddle/Torch）里卷积核默认都是 float32

# 执行卷积
out = conv2d(img, kernel) 

# ========= 绘图=========
f = plt.subplot(121) # 设置画布
f.set_title('input image', fontsize=15) #设置标题
plt.imshow(img, cmap='gray') # cmap 颜色映射表。作用：把矩阵里的数字，转换成屏幕上显示的颜色；数值大 → 白色、数值小 → 黑色


f = plt.subplot(122)
f.set_title('output featuremap', fontsize=15)
plt.imshow(out, cmap='gray')
plt.show()
 #重新开始测试
