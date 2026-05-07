import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 保存 models
def save_models(model,save_path = "models/my_model.pkl"):
    import pickle
    os.makedirs('models', exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"\n你的模型保存至: {save_path}")

# 保存 predictions
def save_predictions(Y_real,Y_pred,Y_proba,save_path = "results/Predictions.csv") :
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df = pd.DataFrame({
        "true_label" : Y_real.flatten(),
        "pred_label" : Y_pred.flatten(),
        "pred_proba" : Y_proba.flatten()
    })
    df.to_csv(save_path,index = False)
    print(f"\nPredictions保存至: {save_path}")

# 保存 Learning-curve 数据
def save_learning_curve(epoch,loss,save_path = "results/Learning_curve.csv") :
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df = pd.DataFrame({
        "epoch" : epoch,
        "loss"  : loss
    })
    df.to_csv(save_path,index = False)
    print(f"\nLearning_curve数据保存至: {save_path}")

# 评估模型
def evaluate(y_pred, y_test):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("-" * 50)
    print("混淆矩阵:")
    print(cm)
    print("-" * 50)
    print(f"准确率 (Accuracy):  {acc:.4f}")
    print(f"精确率 (Precision): {precision:.4f}")
    print(f"召回率 (Recall):    {recall:.4f}")
    print(f"F1分数 (F1-Score):  {f1:.4f}")
    

# 损失曲线
def plot_loss_curve(loss_epochs, loss_values,save_path = "results/figures/Loss_curve.png"):
    plt.plot(loss_epochs, loss_values)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('训练损失曲线')
    plt.grid(True)
    if save_path :
        plt.savefig(save_path,dpi = 100,bbox_inches='tight')
    plt.close()
    print(f"\n损失曲线保存至: {save_path}")

# 二维决策边界
def plot_decision_boundary(model,X,y,save_path = "results/figures/decision_boundary.png") :
    from sklearn.decomposition import PCA
    # 降维
    from config import Config
    if Config.N_FEATURES >= 2 :
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X)
        W = pca.components_.T          # 变换矩阵
        W_pinv = np.linalg.pinv(W)     # 伪逆矩阵
    else :
        raise ValueError("特征纬度太少")

    x_min,x_max = X_2d[:,0].min() - 0.5 ,X_2d[:,0].max() + 0.5                      # 取第一个特征最大最小值作为 x 轴
    y_min,y_max = X_2d[:,1].min() - 0.5 ,X_2d[:,1].max() + 0.5                      # 取第二个特征最大最小值作为 y 轴
    xx,yy = np.meshgrid(np.arange(x_min,x_max,0.01),np.arange(y_min,y_max,0.01))    # 在 x，y 轴每隔0.01取一个值 ；并且把xx，yy作为所有点的横纵坐标
    X_recoverd = np.c_[xx.ravel(),yy.ravel()] @ W_pinv                              # 利用伪逆矩阵，把背景网格扩展到高纬
    pred = model.predict(X_recoverd)
    pred = pred.reshape(xx.shape)                                                   # 把 pred 变成背景网格格式
    plt.contourf(xx, yy, pred, alpha=0.3, cmap='coolwarm')                          # 绘制背景
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c = y, cmap='coolwarm', edgecolors='k')     # 绘制样本点
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.title('决策边界')
    if save_path :
        plt.savefig(save_path,dpi = 100,bbox_inches='tight')
    plt.close()
    print(f"\n二维决策边界图保存至: {save_path}")