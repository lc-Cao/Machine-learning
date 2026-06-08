import numpy as np
from config import Config
from data.make_data_breast_cancer import X, Y, X_train, Y_train, mini_batch_n, X_test, Y_test, X_train_full, X_train_scaled

np.random.seed(Config.SEED)

# 自己的模型
from src.model import Logistic_Regression

n_features = X_train.shape[2]
my_model = Logistic_Regression(n_features)
Epoches = Config.EPOCH
print_size = Epoches // Config.PRINT_SIZE
Loss = [[] for _ in range(2)]

# 训练
print(f"{'训练轮次':<8}{'损失值':<12}")
print("—" * 40)
for epoch in range(Epoches):
    shuffled_index = np.random.permutation(mini_batch_n)
    x_train = X_train[shuffled_index]
    y_train = Y_train[shuffled_index]
    loss = 0.0
    lr = Config.LEARNING_RATE * (0.95 ** (epoch // Config.PRINT_SIZE))
    for batch_i in range(mini_batch_n):
        loss += my_model.train(x_train[batch_i], y_train[batch_i], lr)
    if epoch % print_size == 0:
        Loss[0].append(epoch)
        Loss[1].append(loss / mini_batch_n)
        print(f"{epoch:<12}{loss / mini_batch_n:<12.6f}")

# 预测
my_y_pred = my_model.predict(X_test)
my_y_proba = my_model.forward(X_test)

# 保存数据
from src.utils import save_predictions, save_learning_curve, save_models
save_models(my_model, "models/my_model_breast_cancer.pkl")
save_predictions(Y_test, my_y_pred, my_y_proba, "results/Predictions_breast_cancer.csv")
save_learning_curve(Loss[0], Loss[1], "results/Learning_curve_breast_cancer.csv")

# sklearn模型
from sklearn.linear_model import LogisticRegression

sk_model = LogisticRegression(
    random_state=Config.SEED,
    max_iter=Config.EPOCH
)
sk_model.fit(X_train_scaled, Y_train_full.ravel())
sk_y_pred = sk_model.predict(X_test)

# 模型评估
from src.utils import evaluate
print("\n" + "=" * 50)
print("自己模型 - Breast Cancer")
evaluate(my_y_pred, Y_test)
print("\n" + "=" * 50)
print("sklearn 模型 - Breast Cancer")
evaluate(sk_y_pred, Y_test)

# Loss曲线
from src.utils import plot_loss_curve
plot_loss_curve(Loss[0], Loss[1], "results/figures/Loss_curve_breast_cancer.png")

# 决策边界
from src.utils import plot_decision_boundary
plot_decision_boundary(my_model, X_test, Y_test, "results/figures/decision_boundary_breast_cancer.png")

print("\n乳腺癌数据集实验完成!")