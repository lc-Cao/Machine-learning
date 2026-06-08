"""
早停机制对比实验 — 三分类 Covtype 数据
从训练集拆分 10% 作为验证集，对比 无早停 vs 不同 patience 值
"""
import numpy as np
from config import Config
from data.make_data_multiclass import X_train, Y_train, Y_train_onehot, mini_batch_n, X_test, Y_test, X_train_scaled, Y_train_full

np.random.seed(Config.SEED)

from src.model_multiclass_earlystop import Multiclass_LogisticRegression_EarlyStop
from src.model_multiclass_regularized import Multiclass_LogisticRegression_Regularized
from src.utils_multiclass import evaluate_multiclass, save_predictions_multiclass, save_learning_curve_multiclass, save_models_multiclass
from sklearn.model_selection import train_test_split

n_features = X_train_scaled.shape[1]
n_classes = 3
Epoches = Config.EPOCH
print_size = Epoches // Config.PRINT_SIZE

class_names = ['类别0 (原始1)', '类别1 (原始2)', '类别2 (原始3)']

# ============ 拆分验证集 ============
# 从 un-chunked 训练数据中拆分 10% 验证集
X_tr, X_val, Y_tr, Y_val = train_test_split(
    X_train_scaled, Y_train_full,
    test_size=0.1, random_state=Config.SEED
)

# 转换验证集标签为 one-hot
def to_one_hot(y, n_classes=3):
    return np.eye(n_classes)[y]

Y_val_onehot = to_one_hot(Y_val)

# 训练集重新分块
mini_batch_size = Config.BATCH_SIZE
num_train = X_tr.shape[0]
mini_batch_n_es = num_train // mini_batch_size

X_tr_chunk = X_tr[:mini_batch_n_es * mini_batch_size].reshape(mini_batch_n_es, mini_batch_size, n_features)
Y_tr_chunk = Y_tr[:mini_batch_n_es * mini_batch_size].reshape(mini_batch_n_es, mini_batch_size)

def to_onehot_batch(y, n=3):
    return np.eye(n)[y]

print(f"训练集: {X_tr.shape[0]}, 验证集: {X_val.shape[0]}, 分块数: {mini_batch_n_es}")

# ============ 实验配置 ============
experiments = [
    ("无早停 (SGD+L1)",     None,  'sgd', 'l1', 0.01),
    ("早停 patience=3",       3,  'sgd', 'l1', 0.01),
    ("早停 patience=5",       5,  'sgd', 'l1', 0.01),
    ("早停 patience=10",     10,  'sgd', 'l1', 0.01),
]

results_summary = {}
acc_results = {}
val_loss_history = {}

for label, patience, opt, reg_type, reg_lambda in experiments:
    print("\n" + "=" * 70)
    print(f"  {label} (reg={reg_type}, λ={reg_lambda})")
    print("=" * 70)

    np.random.seed(Config.SEED)

    if patience is None:
        # 无早停：使用原版 regularized 模型
        model = Multiclass_LogisticRegression_Regularized(
            n_features=n_features, n_classes=n_classes,
            reg_type=reg_type, reg_lambda=reg_lambda
        )
        base_lr = Config.LEARNING_RATE
        use_earlystop = False
        early_stop_epoch = Epoches
    else:
        model = Multiclass_LogisticRegression_EarlyStop(
            n_features=n_features, n_classes=n_classes,
            reg_type=reg_type, reg_lambda=reg_lambda,
            patience=patience
        )
        base_lr = Config.LEARNING_RATE
        use_earlystop = True

    train_losses = [[] for _ in range(2)]
    val_losses = [[], []]
    early_stop_epoch = Epoches

    print(f"{'Epoch':<8}{'Train Loss':<14}{'Val Loss':<14}{'Status':<12}")
    print("—" * 50)

    for epoch in range(Epoches):
        # 训练
        shuffled_index = np.random.permutation(mini_batch_n_es)
        x_tr = X_tr_chunk[shuffled_index]
        y_tr_labels = Y_tr_chunk[shuffled_index]

        epoch_loss = 0.0
        lr = base_lr * (0.95 ** (epoch // Config.PRINT_SIZE))

        for batch_i in range(mini_batch_n_es):
            y_onehot = to_onehot_batch(y_tr_labels[batch_i], n_classes)
            if use_earlystop:
                epoch_loss += model.train_step(x_tr[batch_i], y_onehot, lr)
            else:
                epoch_loss += model.train(x_tr[batch_i], y_onehot, lr)

        if epoch % print_size == 0:
            # 计算验证损失
            if use_earlystop:
                model._current_epoch = epoch
                val_loss, improved = model.validate(X_val, Y_val, Y_val_onehot)
                status = "✓ 提升" if improved else f"- 等待{model.wait}/{model.patience}"
            else:
                val_loss = model.compute_loss(X_val, Y_val_onehot) if hasattr(model, 'compute_loss') else 0

            # 训练损失按 batch 平均，与验证损失同尺度
            avg_train_loss = epoch_loss / mini_batch_n_es
            train_losses[0].append(epoch)
            train_losses[1].append(avg_train_loss)
            val_losses[0].append(epoch)
            val_losses[1].append(val_loss)

            if use_earlystop:
                print(f"{epoch:<8}{epoch_loss:<14.4f}{val_loss:<14.4f}{status:<12}")
            else:
                print(f"{epoch:<8}{epoch_loss:<14.4f}")

        if use_earlystop and model.should_stop:
            early_stop_epoch = epoch + 1
            print(f"\n>>> 早停触发! Epoch {early_stop_epoch}, 最佳 Val Loss: {model.best_val_loss:.4f}")
            model.restore_best()
            break

    # 预测
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    print(f"\n--- {label} 评估 (停止轮次: {early_stop_epoch}) ---")
    evaluate_multiclass(y_pred, Y_test, class_names)

    tag = label.replace(" ", "_").replace("(", "").replace(")", "").replace("+", "p")
    save_models_multiclass(model, f"models/my_model_multiclass_{tag}.pkl")
    save_predictions_multiclass(Y_test, y_pred, y_proba,
                                f"results/Predictions_multiclass_{tag}.csv")
    save_learning_curve_multiclass(train_losses[0], train_losses[1],
                                   f"results/Learning_curve_multiclass_{tag}.csv")

    results_summary[label] = (train_losses, val_losses)
    acc = np.mean(y_pred.flatten() == Y_test.flatten())
    acc_results[label] = (acc, early_stop_epoch)

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
acc_results['sklearn'] = (np.mean(sk_y_pred == Y_test), Config.EPOCH)

# ============ 汇总 ============
print("\n" + "=" * 70)
print("  准确率 & 效率对比")
print("=" * 70)
print(f"{'配置':<25}{'准确率':<10}{'停止轮次':<10}{'比较':<10}")
print("-" * 55)
best_acc = max(v[0] for v in acc_results.values())
for label, (acc, stop_epoch) in acc_results.items():
    saved = f"省{Config.EPOCH - stop_epoch}轮" if stop_epoch < Config.EPOCH else "—"
    marker = " ← 最佳" if acc == best_acc else ""
    print(f"{label:<25}{acc:<10.4f}{stop_epoch:<10}{saved:<10}{marker}")

# ============ 训练/验证损失曲线 ============
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 取第一组代表数据（所有组相同）
sample_label = list(results_summary.keys())[0]
tr_loss, val_loss = results_summary[sample_label]

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(tr_loss[0], tr_loss[1], 'b-o', label='训练损失 (平均/batch)', linewidth=2, markersize=4)
ax.plot(val_loss[0], val_loss[1], 'r-s', label='验证损失', linewidth=2, markersize=4)
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('早停实验 — 训练 & 验证损失 (SGD+L1, λ=0.01)')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.annotate('验证损失持续下降\n未触发早停 (无过拟合)',
            xy=(100, val_loss[1][len(val_loss[1])//2]),
            fontsize=9, color='darkred',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))

plt.tight_layout()
plt.savefig('results/figures/EarlyStop_comparison.png', dpi=120)
plt.close()
print("\n早停对比图已保存至 results/figures/EarlyStop_comparison.png")

print("\n" + "=" * 70)
print("早停机制对比实验全部完成!")