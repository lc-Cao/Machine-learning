import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 保存 models
def save_models_multiclass(model, save_path="models/my_model_multiclass.pkl"):
    import pickle
    os.makedirs('models', exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"\n你的多分类模型保存至: {save_path}")

# 保存 predictions
def save_predictions_multiclass(Y_real, Y_pred, Y_proba, save_path="results/Predictions_multiclass.csv"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Y_proba 形状为 (n_samples, n_classes)，需要展开为多列
    n_classes = Y_proba.shape[1]
    proba_columns = {}
    for i in range(n_classes):
        proba_columns[f'prob_class_{i}'] = Y_proba[:, i]
    
    df = pd.DataFrame({
        "true_label": Y_real.flatten(),
        "pred_label": Y_pred.flatten(),
        **proba_columns
    })
    df.to_csv(save_path, index=False)
    print(f"\n多分类 Predictions保存至: {save_path}")

# 保存 Learning-curve 数据
def save_learning_curve_multiclass(epoch, loss, save_path="results/Learning_curve_multiclass.csv"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df = pd.DataFrame({
        "epoch": epoch,
        "loss": loss
    })
    df.to_csv(save_path, index=False)
    print(f"\n多分类 Learning_curve数据保存至: {save_path}")

# 评估多分类模型
def evaluate_multiclass(y_pred, y_test, class_names=None):
    """
    评估多分类模型性能
    Args:
        y_pred: 预测标签
        y_test: 真实标签
        class_names: 类别名称列表，如 ['类别0', '类别1', '类别2']
    """
    acc = accuracy_score(y_test, y_pred)
    
    # 计算宏平均和加权平均指标
    precision_macro = precision_score(y_test, y_pred, average='macro')
    recall_macro = recall_score(y_test, y_pred, average='macro')
    f1_macro = f1_score(y_test, y_pred, average='macro')
    
    precision_weighted = precision_score(y_test, y_pred, average='weighted')
    recall_weighted = recall_score(y_test, y_pred, average='weighted')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    
    cm = confusion_matrix(y_test, y_pred)
    
    print("=" * 60)
    print("混淆矩阵:")
    print(cm)
    print("=" * 60)
    print(f"准确率 (Accuracy): {acc:.4f}")
    print("\n宏平均 (Macro Average):")
    print(f"  精确率 (Precision): {precision_macro:.4f}")
    print(f"  召回率 (Recall):    {recall_macro:.4f}")
    print(f"  F1分数 (F1-Score):  {f1_macro:.4f}")
    print("\n加权平均 (Weighted Average):")
    print(f"  精确率 (Precision): {precision_weighted:.4f}")
    print(f"  召回率 (Recall):    {recall_weighted:.4f}")
    print(f"  F1分数 (F1-Score):  {f1_weighted:.4f}")
    
    # 打印详细分类报告
    print("\n" + "=" * 60)
    print("详细分类报告:")
    if class_names is not None:
        print(classification_report(y_test, y_pred, target_names=class_names))
    else:
        print(classification_report(y_test, y_pred))

# 损失曲线（与二分类相同）
def plot_loss_curve_multiclass(loss_epochs, loss_values, save_path="results/figures/Loss_curve_multiclass.png"):
    plt.figure(figsize=(10, 6))
    plt.plot(loss_epochs, loss_values)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('多分类训练损失曲线')
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"\n多分类损失曲线保存至: {save_path}")

# 多分类决策边界可视化
def plot_decision_boundary_multiclass(model, X, y, save_path="results/figures/decision_boundary_multiclass.png"):
    """
    绘制多分类决策边界（使用PCA降维到2维）
    Args:
        model: 多分类模型，需要有 predict 方法
        X: 特征数据，形状 (n_samples, n_features)
        y: 真实标签，形状 (n_samples,)
    """
    from sklearn.decomposition import PCA
    
    if X.shape[1] < 2:
        raise ValueError("特征纬度太少，无法进行降维可视化")
    
    # 使用PCA降维到2维
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)
    
    # 创建网格数据
    x_min, x_max = X_2d[:, 0].min() - 0.5, X_2d[:, 0].max() + 0.5
    y_min, y_max = X_2d[:, 1].min() - 0.5, X_2d[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
    
    # 将网格点转换回原始特征空间
    W = pca.components_.T
    W_pinv = np.linalg.pinv(W)
    X_recovered = np.c_[xx.ravel(), yy.ravel()] @ W_pinv
    
    # 预测网格点的类别
    pred = model.predict(X_recovered)
    pred = pred.reshape(xx.shape)
    
    # 绘制决策边界
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, pred, alpha=0.3, cmap='viridis')
    
    # 绘制真实数据点
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap='viridis', edgecolors='k', s=50)
    
    # 添加图例
    unique_classes = np.unique(y)
    legend_elements = []
    for cls in unique_classes:
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                          markerfacecolor=plt.cm.viridis(cls/max(unique_classes)), 
                                          markersize=10, label=f'类别 {cls}'))
    
    plt.xlabel('主成分 1')
    plt.ylabel('主成分 2')
    plt.title('多分类决策边界')
    plt.legend(handles=legend_elements, title='真实类别')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"\n多分类决策边界图保存至: {save_path}")