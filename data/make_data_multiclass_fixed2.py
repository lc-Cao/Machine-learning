import numpy as np
from config import Config
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

np.random.seed(Config.SEED)

"""
数据来源：使用 sklearn 的 fetch_covtype 数据集 
- 特征数: 54 (自动获取)
- 原始类别数: 7 (森林覆盖类型)
- 样本数: 581,012
- 随机种子: Config.SEED
- 我们将其转换为三分类问题：使用类别1,2,3（样本量最大的三个类别）

数据处理：
- 标准化：使用 StandardScaler 统一特征量纲 
- 数据划分：按 Config.TEST_RATE 划分测试集 
- 数据分块：将训练集整理为 (批次数, 批大小, 特征数) 格式 
- 标签处理：转换为 one-hot 编码格式
"""

# 1. 数据加载 
print("正在下载covtype数据集（可能需要几分钟）...")
data = fetch_covtype(download_if_missing=True)
X = data.data
Y = data.target

# 转换为三分类问题：使用类别1,2,3（样本量最大的三个类别）
mask = (Y == 1) | (Y == 2) | (Y == 3)
X = X[mask]
Y = Y[mask]

# 将标签转换为0,1,2 (1变为0, 2变为1, 3变为2)
label_mapping = {1: 0, 2: 1, 3: 2}
Y_mapped = np.array([label_mapping[y] for y in Y])

# 为了加快训练速度，我们采样一部分数据
# 使用45000个样本以保持类别平衡
np.random.seed(Config.SEED)
sample_size = 45000  # 采样45000个样本
if len(X) > sample_size:
    indices = np.random.choice(len(X), sample_size, replace=False)
    X = X[indices]
    Y_mapped = Y_mapped[indices]

# 统计类别分布
unique, counts = np.unique(Y_mapped, return_counts=True)
print(f"\n三分类数据集类别分布:")
for cls, cnt in zip(unique, counts):
    percentage = cnt / len(Y_mapped) * 100
    print(f"  类别 {cls}: {cnt} 个样本 ({percentage:.1f}%)")

# 2. 数据划分 
X_train_full, X_test_raw, Y_train_full, Y_test = train_test_split(
    X, Y_mapped, 
    test_size=Config.TEST_RATE, 
    random_state=Config.SEED
)

# 3. 特征标准化 (真实数据必做步骤) 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_full) # 训练集标准化
X_test = scaler.transform(X_test_raw)               # 测试集使用同样的缩放标准

# 4. 动态计算分块逻辑 
mini_batch_size = Config.BATCH_SIZE
# 重新获取当前训练集的总数和特征数
num_train_samples = X_train_scaled.shape[0]
n_features = X_train_scaled.shape[1]

mini_batch_n = num_train_samples // mini_batch_size # 计算完整数据块数量

# 截断多余的数据，确保可以被 reshape
X_train_truncated = X_train_scaled[:mini_batch_n * mini_batch_size]
Y_train_truncated = Y_train_full[:mini_batch_n * mini_batch_size]

# 5. 数据分块整理 
# 整理为 (批次数, 批大小, 特征数)
X_train = X_train_truncated.reshape(mini_batch_n, mini_batch_size, n_features)
# 整理标签为 (批次数, 批大小) - 不需要额外的维度用于多分类
Y_train = Y_train_truncated.reshape(mini_batch_n, mini_batch_size)

# 6. 转换为 one-hot 编码（用于训练）
def to_one_hot(y, num_classes=3):
    """将整数标签转换为 one-hot 编码"""
    return np.eye(num_classes)[y]

# 创建 one-hot 编码的训练标签
Y_train_onehot = to_one_hot(Y_train_truncated).reshape(mini_batch_n, mini_batch_size, 3)

print(f"\n数据集加载完成：Covtype (类别1,2,3三分类)")
print(f"原始样本数: {len(data.data)}, 采样样本数: {len(X)}")
print(f"训练分块形状 X: {X_train.shape}") # (Batch_N, Batch_Size, Features)
print(f"训练分块形状 Y (整数标签): {Y_train.shape}") # (Batch_N, Batch_Size)
print(f"训练分块形状 Y (one-hot): {Y_train_onehot.shape}") # (Batch_N, Batch_Size, 3)

# 导出所有需要的变量
__all__ = ['X_train', 'Y_train', 'Y_train_onehot', 'mini_batch_n', 'X_test', 'Y_test', 
           'X_train_scaled', 'Y_train_full', 'X_train_truncated', 'Y_train_truncated']

# 确保变量可以被正确导入
if __name__ != "__main__":
    # 当作为模块导入时，确保这些变量可用
    pass