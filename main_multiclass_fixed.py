import numpy as np
from config import Config
from data.make_data_multiclass import X_train, Y_train, Y_train_onehot, mini_batch_n, X_test, Y_test

np.random.seed(Config.SEED)

# 自己的多分类模型
from src.model_multiclass import Multiclass_LogisticRegression

n_features = X_train.shape[2] 
n_classes = 3
my_model = Multiclass_LogisticRegression(n_features, n_classes)

Epoches = Config.EPOCH                                                  # 训练轮次
print_size = Epoches // Config.PRINT_SIZE                               # 打印次数
Loss = [[] for _ in range(2)]                                           # 记录损失

# 训练
print(f"{'训练轮次':<8}{'损失值':<12}")
print("—" * 40)
for epoch in range(Epoches):
    # 随机打乱数据块
    shuffled_index = np.random.permutation(mini_batch_n)
    x_train = X_train[shuffled_index]
    y_train_onehot = Y_train_onehot[shuffled_index]
    y_train_labels = Y_train[shuffled_index]
    
    loss = 0.0
    lr = Config.LEARNING_RATE * (0.95 ** (epoch // Config.PRINT_SIZE))  # 学习率衰减
    
    # 分块训练数据
    for batch_i in range(mini_batch_n):
        loss += my_model.train(x_train[batch_i], y_train_onehot[batch_i], lr)
    
    if epoch % print_size == 0:
        Loss[0].append(epoch)
        Loss[1].append(loss)
        print(f"{epoch:<12}{loss:<12.6f}")

# 预测
my_y_pred = my_model.predict(X_test)
my_y_proba = my_model.predict_proba(X_test)

# 保存数据
from utils_multiclass import save_predictions_multiclass, save_learning_curve_multiclass, save_models_multiclass
save_models_multiclass(my_model)
save_predictions_multiclass(Y_test, my_y_pred, my_y_proba)
save_learning_curve_multiclass(Loss[0], Loss[1])

# sklearn多分类模型
from sklearn.linear_model import LogisticRegression

# 注意：需要重新获取完整的训练数据，因为Y_train_full在make_data_multiclass中没有导出
# 我们将从data模块重新导入
from data.make_data_multiclass import X_train_scaled, Y_train_full

# 使用正确的参数名，sklearn 1.8.0版本中multi_class参数可能已改名或移除
# 尝试使用默认参数
sk_model = LogisticRegression(
    random_state=Config.SEED,                                         # 随机种子
    max_iter=Config.EPOCH,                                            # 训练轮次
    solver='lbfgs'                                                    # lbfgs适用于多分类
)
# 训练
sk_model.fit(X_train_scaled, Y_train_full)
# 测试
sk_y_pred = sk_model.predict(X_test)
sk_y_proba = sk_model.predict_proba(X_test)

# 模型评估
from utils_multiclass import evaluate_multiclass, plot_loss_curve_multiclass, plot_decision_boundary_multiclass

print("\n" + "=" * 60)
print("自己模型 (多分类)")
class_names = ['类别0 (原始1)', '类别1 (原始2)', '类别2 (原始3)']
evaluate_multiclass(my_y_pred, Y_test, class_names)

print("\n" + "=" * 60)
print("sklearn 模型 (多分类)")
evaluate_multiclass(sk_y_pred, Y_test, class_names)

# Loss曲线
plot_loss_curve_multiclass(Loss[0], Loss[1])

# 二维决策边界
try:
    plot_decision_boundary_multiclass(my_model, X_test, Y_test)
except Exception as e:
    print(f"\n绘制决策边界时出错: {e}")
    print("可能是因为样本数量或特征维度问题，跳过决策边界绘制。")

# 将sklearn预测结果也保存下来
print("\n" + "=" * 60)
print("保存sklearn模型的预测结果...")
from utils_multiclass import save_predictions_multiclass as save_sk_predictions
save_sk_predictions(Y_test, sk_y_pred, sk_y_proba, "results/Predictions_sklearn_multiclass.csv")

print("\n" + "=" * 60)
print("多分类实验完成!")
print(f"  样本总数: {len(Y_test)}")
print(f"  特征维度: {n_features}")
print(f"  类别数量: {n_classes}")
print(f"  训练轮次: {Epoches}")
print("结果已保存到 results/ 目录下")