import numpy as np

# 1. 构造一个 3D 数据集
np.random.seed(0)
X = np.dot(np.random.rand(3,3), np.random.randn(3,100)).T  # [100, 3]
print("原始数据维度:", X.shape)

# 2. 数据中心化
X_centered = X - X.mean(axis=0)

# 3. 计算协方差矩阵
cov = np.cov(X_centered, rowvar=False)

# 4. 求特征值和特征向量
eig_vals, eig_vecs = np.linalg.eigh(cov)

# 5. 按特征值大小排序
idx = np.argsort(eig_vals)[::-1]
eig_vals = eig_vals[idx]
eig_vecs = eig_vecs[:, idx]

# 6. 计算解释方差比例
explained_var_ratio = eig_vals / eig_vals.sum()
cum_explained_var = np.cumsum(explained_var_ratio)

print("\n每个主成分的解释方差比例:")
for i, ratio in enumerate(explained_var_ratio):
    print(f"PC{i+1}: {ratio:.4f} ({cum_explained_var[i]:.4f} 累计)")

# 7. 取前2个主成分做降维
W = eig_vecs[:, :2]
X_pca = X_centered @ W
print("\n降维后数据维度:", X_pca.shape)

# 输出前5个样本对比
print("\n原始数据(前5个样本，3维):\n", X[:5])
print("\n降维后数据(前5个样本，2维):\n", X_pca[:5])