from sklearn.datasets import fetch_covtype
import numpy as np

# 加载数据集
print("正在下载covtype数据集...")
data = fetch_covtype(download_if_missing=True)
X = data.data
Y = data.target

print(f"总样本数: {len(Y)}")
print(f"特征数: {X.shape[1]}")

# 类别分布分析
unique, counts = np.unique(Y, return_counts=True)
print("\n类别分布:")
for cls, cnt in zip(unique, counts):
    percentage = cnt / len(Y) * 100
    print(f"  类别 {cls}: {cnt} 个样本 ({percentage:.1f}%)")

# 检查最合适的三个类别进行三分类实验
# 选择样本量最大的三个类别
sorted_indices = np.argsort(counts)[::-1]  # 从大到小排序
print("\n样本量最大的三个类别:")
for i in range(min(3, len(unique))):
    idx = sorted_indices[i]
    cls = unique[idx]
    cnt = counts[idx]
    percentage = cnt / len(Y) * 100
    print(f"  {i+1}. 类别 {cls}: {cnt} 个样本 ({percentage:.1f}%)")

# 验证这些类别是否具有足够的数据量进行实验
selected_classes = unique[sorted_indices[:3]]
print(f"\n建议用于三分类实验的类别: {selected_classes}")
print("类别映射: 类别1->0, 类别2->1, 类别3->2")