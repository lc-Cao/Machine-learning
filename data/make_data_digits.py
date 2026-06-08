import numpy as np
from config import Config
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

np.random.seed(Config.SEED)

"""
数据来源：使用 sklearn 的 load_digits 数据集
- 特征数: 64 (8x8 像素)
- 原始类别数: 10 (数字 0-9)
- 样本数: 1,797
- 随机种子: Config.SEED
- 转换为二分类问题：只使用类别 0 和 1

数据处理：
- 标准化：使用 StandardScaler 统一特征量纲
- 数据划分：按 Config.TEST_RATE 划分测试集
- 数据分块：将训练集整理为 (批次数, 批大小, 特征数) 格式
"""

# 1. 数据加载
print("正在加载 Digits 手写数字数据集...")
data = load_digits()
X = data.data
Y = data.target

# 转换为二分类问题：只使用类别 0 和 1
mask = (Y == 0) | (Y == 1)
X = X[mask]
Y = Y[mask].reshape(-1, 1)

print(f"数据集加载完成：Digits (类别0/1 二分类)")
print(f"样本数: {len(X)}, 特征数: {X.shape[1]}")
print(f"类别分布: 类别0: {np.sum(Y==0)}, 类别1: {np.sum(Y==1)}")

# 2. 数据划分
X_train_full, X_test_raw, Y_train_full, Y_test = train_test_split(
    X, Y,
    test_size=Config.TEST_RATE,
    random_state=Config.SEED
)

# 3. 特征标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_full)
X_test = scaler.transform(X_test_raw)

# 4. 动态计算分块逻辑
mini_batch_size = Config.BATCH_SIZE
num_train_samples = X_train_scaled.shape[0]
n_features = X_train_scaled.shape[1]

mini_batch_n = num_train_samples // mini_batch_size

# 截断多余的数据
X_train_truncated = X_train_scaled[:mini_batch_n * mini_batch_size]
Y_train_truncated = Y_train_full[:mini_batch_n * mini_batch_size]

# 5. 数据分块整理
X_train = X_train_truncated.reshape(mini_batch_n, mini_batch_size, n_features)
Y_train = Y_train_truncated.reshape(mini_batch_n, mini_batch_size, 1)

print(f"训练分块形状: {X_train.shape}")
print(f"测试集大小: {len(X_test)}")