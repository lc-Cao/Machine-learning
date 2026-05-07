import numpy as np
from config import Config
from data.make_data import X,Y,X_train,Y_train,mini_batch_n,X_test,Y_test,X_train_full,Y_train_full,X_train_scaled

np.random.seed(Config.SEED)   


# 自己的模型
from src.model import Logistic_Regression

n_features = X_train.shape[2] 
my_model = Logistic_Regression(n_features)
Epoches = Config.EPOCH                                                  # 训练轮次
print_size = Epoches // Config.PRINT_SIZE                               # 打印次数
Loss = [[] for _ in range(2)]                                           # 记录损失

  # 训练
print(f"{'训练轮次':<8}{'损失值':<12}")
print("—" * 40)
for epoch in range(Epoches) :
    # 随机打乱数据块
    shuffled_index = np.random.permutation(mini_batch_n)
    x_train = X_train[shuffled_index]
    y_train = Y_train[shuffled_index]
    loss = 0.0
    lr = Config.LEARNING_RATE * (0.95 ** (epoch // Config.PRINT_SIZE))  # 学习率衰减
    # 分块训练数据
    for batch_i in range(mini_batch_n) :
        loss += my_model.train(x_train[batch_i],y_train[batch_i],lr)
    if epoch % print_size == 0 :
        Loss[0].append(epoch)
        Loss[1].append(loss)
        print(f"{epoch:<12}{loss:<12.6f}")

  # 预测
my_y_pred = my_model.predict(X_test)
my_y_proba = my_model.forward(X_test)

# 保存数据
from src.utils import save_predictions,save_learning_curve,save_models
save_models(my_model)
save_predictions(Y_test,my_y_pred,my_y_proba)
save_learning_curve(Loss[0],Loss[1])

# sklearn模型
from sklearn.linear_model import LogisticRegression

sk_model = LogisticRegression(
    random_state = Config.SEED,                                         # 随机种子
    max_iter = Config.EPOCH                                             # 训练轮次
)
  # 训练
sk_model.fit(X_train_scaled, Y_train_full.ravel())   
  # 测试
sk_y_pred = sk_model.predict(X_test)


# 模型评估
from src.utils import evaluate
print("\n自己模型")
evaluate(my_y_pred,Y_test)
print("\n" + "=" * 50 + "\n sk 模型")
evaluate(sk_y_pred,Y_test)

# Loss曲线
from src.utils import plot_loss_curve
plot_loss_curve(Loss[0],Loss[1])

# 二维决策边界
from src.utils import plot_decision_boundary
plot_decision_boundary(my_model,X_test,Y_test)