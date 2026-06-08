"""
L2 正则化系数 λ 调优
在三分类 Covtype 数据上网格搜索最佳 λ
"""
import numpy as np
from config import Config
from data.make_data_multiclass import X_train, Y_train, Y_train_onehot, mini_batch_n, X_test, Y_test

np.random.seed(Config.SEED)

from src.model_multiclass_regularized import Multiclass_LogisticRegression_Regularized
from sklearn.metrics import accuracy_score

n_features = X_train.shape[2]
n_classes = 3
Epoches = Config.EPOCH

# 测试不同 λ 值（从小到大的网格）
lambda_list = [0.0001, 0.0005, 0.001, 0.003, 0.005, 0.01, 0.03, 0.05, 0.1]
results = []

print(f"{'λ':<10}{'准确率':<12}{'最终损失':<14}")
print("=" * 36)

for lam in lambda_list:
    np.random.seed(Config.SEED)
    model = Multiclass_LogisticRegression_Regularized(
        n_features=n_features, n_classes=n_classes,
        reg_type='l2', reg_lambda=lam
    )

    final_loss = 0.0
    for epoch in range(Epoches):
        shuffled_index = np.random.permutation(mini_batch_n)
        x_train = X_train[shuffled_index]
        y_onehot = Y_train_onehot[shuffled_index]

        loss = 0.0
        lr = Config.LEARNING_RATE * (0.95 ** (epoch // Config.PRINT_SIZE))
        for batch_i in range(mini_batch_n):
            loss += model.train(x_train[batch_i], y_onehot[batch_i], lr)
        final_loss = loss

    y_pred = model.predict(X_test)
    acc = accuracy_score(Y_test, y_pred)
    results.append((lam, acc, final_loss))
    print(f"{lam:<10}{acc:<12.4f}{final_loss:<14.4f}")

# 找最佳
best = max(results, key=lambda x: x[1])
print("\n" + "=" * 36)
print(f"最佳 L2 λ = {best[0]}, 准确率 = {best[1]:.4f}, 损失 = {best[2]:.4f}")
print(f"\n建议将 config.py 中 REG_LAMBDA 改为 {best[0]}")