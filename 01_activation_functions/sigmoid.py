# encoding=utf-8

import matplotlib.pylab as plt
import numpy as np


# 定义 Sigmoid 激活函数
# 数学公式：f(x) = 1 / (1 + e^(-x))
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# 测试单个数值的输出（以圆周率 π 为例）
# np.exp(-3.1415926) 约为 0.0432，1 / 1.0432 约为 0.958
val_pi = 3.1415926
print(f"当 x = {val_pi} 时，Sigmoid 输出值(保留两位小数): {round(sigmoid(val_pi), 2)}")


# --- 开始绘制 Sigmoid 函数图像 ---

# 1. 生成自变量 X：从 -5.0 到 5.0，步长为 0.1 的等差数列（不包含 5.0）
X = np.arange(-5.0, 5.0, 0.1)

# 2. 计算因变量 Y：利用 NumPy 的广播机制，批量计算每个 X 对应的 Sigmoid 值
Y = sigmoid(X)

# 3. 绘制折线图
plt.plot(X, Y, label="Sigmoid", color="blue")

# 4. 美化图表设置
plt.title("Sigmoid Activation Function")  # 添加图表标题
plt.xlabel("Input (x)")  # 添加 X 轴标签
plt.ylabel("Output (y)")  # 添加 Y 轴标签
plt.grid(True, linestyle="--", alpha=0.6)  # 添加网格线，方便观察曲线走向
plt.axhline(0.5, color="red", linestyle=":")  # 额外画一条 y=0.5 的参考线（Sigmoid 的中心对称点）

# 5. 设置 Y 轴的显示范围（上下留出 0.1 的边距，视觉效果更好）
plt.ylim(-0.1, 1.1)

# 6. 显示图例并渲染图像
plt.legend()
plt.show()