"""
Adam 优化器对比实验 — 三分类 Covtype 数据
对比：SGD / Adam，每种搭配 无正则化 / L1 正则化
"""
import numpy as np
from config import Config
from data.make_data_multiclass import X_train, Y_train, Y_train_onehot, mini_batch_n, X_test, Y_test, X_train_scaled, Y_train_full

np.random.seed(Config.SEED)

from src.model_multiclass_regularized import Multiclass_LogisticRegression_Regularized
from src.model_multiclass_adam import Multiclass_LogisticRegression_Adam
from src.utils_multiclass import evaluate_multiclass, save_predictions_multiclass, save_learning_curve_multiclass, save_models_multiclass

n_features = X_train.shape[2]
n_classes = 3
Epoches = Config.EPOCH
print_size = Epoches // Config.PRINT_SIZE

class_names = ['类别0 (原始1)', '类别1 (原始2)', '类别2 (原始3)']

# ============ 实验配置 ============
experiments = [
    ("SGD 无正则化",  'sgd', 'none', 0.0),
    ("SGD + L1",       'sgd', 'l1',   0.01),
    ("Adam 无正则化",  'adam','none', 0.0),
    ("Adam + L1",      'adam','l1',   0.01),
]

results_summary = {}
acc_results = {}

for label, opt, reg_type, reg_lambda in experiments:
    print("\n" + "=" * 70)
    print(f"  {label} (reg={reg_type}, λ={reg_lambda})")
    print("=" * 70)

    np.random.seed(Config.SEED)

    if opt == 'sgd':
        model = Multiclass_LogisticRegression_Regularized(
            n_features=n_features, n_classes=n_classes,
            reg_type=reg_type, reg_lambda=reg_lambda
        )
        base_lr = Config.LEARNING_RATE
    else:  # adam
        model = Multiclass_LogisticRegression_Adam(
            n_features=n_features, n_classes=n_classes,
            reg_type=reg_type, reg_lambda=reg_lambda,
            lr=0.001  # Adam 推荐学习率
        )
        base_lr = 0.001

    Loss = [[] for _ in range(2)]

    # 训练
    print(f"{'训练轮次':<8}{'损失值':<14}")
    print("—" * 40)
    for epoch in range(Epoches):
        shuffled_index = np.random.permutation(mini_batch_n)
        x_train = X_train[shuffled_index]
        y_train_onehot = Y_train_onehot[shuffled_index]

        loss = 0.0
        lr = base_lr * (0.95 ** (epoch // Config.PRINT_SIZE))
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
    tag = label.replace(" ", "_").replace("+", "p")
    save_models_multiclass(model, f"models/my_model_multiclass_{tag}.pkl")
    save_predictions_multiclass(Y_test, y_pred, y_proba,
                                f"results/Predictions_multiclass_{tag}.csv")
    save_learning_curve_multiclass(Loss[0], Loss[1],
                                   f"results/Learning_curve_multiclass_{tag}.csv")

    results_summary[label] = Loss
    acc = np.mean(y_pred.flatten() == Y_test.flatten())
    acc_results[label] = acc

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
acc_results['sklearn'] = np.mean(sk_y_pred == Y_test)

# ============ 准确率对比表 ============
print("\n" + "=" * 70)
print("  准确率对比汇总")
print("=" * 70)
print(f"{'配置':<20}{'准确率':<12}")
print("-" * 32)
for label, acc in acc_results.items():
    marker = " ← 最佳" if acc == max(acc_results.values()) else ""
    print(f"{label:<20}{acc:<12.4f}{marker}")

# ============ 损失曲线对比 ============
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# 左图：损失曲线
colors = {'SGD 无正则化': '#1f77b4', 'SGD + L1': '#ff7f0e',
          'Adam 无正则化': '#2ca02c', 'Adam + L1': '#d62728'}
for label, (epochs, losses) in results_summary.items():
    ax1.plot(epochs, losses, label=label, color=colors.get(label), linewidth=2)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('SGD vs Adam — 训练损失曲线')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 右图：准确率柱状图
labels = list(acc_results.keys())
values = list(acc_results.values())
bars = ax2.bar(labels, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
ax2.set_ylabel('Accuracy')
ax2.set_title('准确率对比')
ax2.set_ylim(min(values) * 0.995, max(values) * 1.002)
ax2.tick_params(axis='x', rotation=15)
for bar, val in zip(bars, values):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0002,
             f'{val:.4f}', ha='center', va='bottom', fontsize=9)
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('results/figures/Adam_comparison.png', dpi=120)
plt.close()
print("\nAdam 对比图已保存至 results/figures/Adam_comparison.png")

print("\n" + "=" * 70)
print("Adam 优化器对比实验全部完成!")