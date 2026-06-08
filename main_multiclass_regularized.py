"""
正则化对比实验 — 三分类 Covtype 数据
测试：无正则化 / L1 正则化 / L2 正则化 三种配置
"""
import numpy as np
from config import Config
from data.make_data_multiclass import X_train, Y_train, Y_train_onehot, mini_batch_n, X_test, Y_test, X_train_scaled, Y_train_full

np.random.seed(Config.SEED)

from src.model_multiclass_regularized import Multiclass_LogisticRegression_Regularized
from src.utils_multiclass import evaluate_multiclass, save_predictions_multiclass, save_learning_curve_multiclass, save_models_multiclass

n_features = X_train.shape[2]
n_classes = 3
Epoches = Config.EPOCH
print_size = Epoches // Config.PRINT_SIZE

class_names = ['类别0 (原始1)', '类别1 (原始2)', '类别2 (原始3)']

# ============ 实验配置 ============
experiments = [
    ("无正则化",  'none', 0.0),
    ("L1 正则化", 'l1',   0.01),     # L1 最佳 λ
    ("L2 正则化", 'l2',   0.0001),    # L2 最佳 λ
]

results_summary = {}

for label, reg_type, reg_lambda in experiments:
    print("\n" + "=" * 70)
    print(f"  {label} (λ={reg_lambda})")
    print("=" * 70)

    # 重新初始化模型（相同随机种子保证可比性）
    np.random.seed(Config.SEED)
    model = Multiclass_LogisticRegression_Regularized(
        n_features=n_features, n_classes=n_classes,
        reg_type=reg_type, reg_lambda=reg_lambda
    )

    Loss = [[] for _ in range(2)]

    # 训练
    print(f"{'训练轮次':<8}{'损失值':<14}")
    print("—" * 40)
    for epoch in range(Epoches):
        shuffled_index = np.random.permutation(mini_batch_n)
        x_train = X_train[shuffled_index]
        y_train_onehot = Y_train_onehot[shuffled_index]

        loss = 0.0
        lr = Config.LEARNING_RATE * (0.95 ** (epoch // Config.PRINT_SIZE))
        for batch_i in range(mini_batch_n):
            loss += model.train(x_train[batch_i], y_train_onehot[batch_i], lr)

        if epoch % print_size == 0:
            Loss[0].append(epoch)
            Loss[1].append(loss)
            print(f"{epoch:<12}{loss:<14.6f}")

    # 预测
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    # 评估
    print(f"\n--- {label} 评估 ---")
    evaluate_multiclass(y_pred, Y_test, class_names)

    # 保存
    tag = label.replace("正则化", "").replace(" ", "").replace("无", "none")
    save_models_multiclass(model, f"models/my_model_multiclass_{tag}.pkl")
    save_predictions_multiclass(Y_test, y_pred, y_proba,
                                f"results/Predictions_multiclass_{tag}.csv")
    save_learning_curve_multiclass(Loss[0], Loss[1],
                                   f"results/Learning_curve_multiclass_{tag}.csv")

    results_summary[label] = Loss

# ============ sklearn 基准 ============
from sklearn.linear_model import LogisticRegression

print("\n" + "=" * 70)
print("  sklearn 基准（默认 L2）")
print("=" * 70)

sk_model = LogisticRegression(
    random_state=Config.SEED, max_iter=Config.EPOCH,
    solver='lbfgs'
)
sk_model.fit(X_train_scaled, Y_train_full)
sk_y_pred = sk_model.predict(X_test)
evaluate_multiclass(sk_y_pred, Y_test, class_names)

# ============ 对比损失曲线 ============
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

plt.figure(figsize=(10, 6))
for label, (epochs, losses) in results_summary.items():
    plt.plot(epochs, losses, label=label, linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('正则化对比 — 训练损失曲线')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/figures/Loss_curve_regularization_comparison.png', dpi=100)
plt.close()
print("\n正则化对比损失曲线已保存至 results/figures/Loss_curve_regularization_comparison.png")

print("\n" + "=" * 70)
print("正则化对比实验全部完成!")