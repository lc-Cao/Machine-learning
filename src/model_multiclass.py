import numpy as np
from config import Config

def softmax(x):
    """
    计算 softmax 函数
    输入: x 形状为 (batch_size, n_classes) 或 (batch_size, n_classes, 1)
    输出: 每个样本在每个类别上的概率，形状与输入相同
    """
    # 数值稳定性优化：减去最大值
    x = np.clip(x, -500, 500)  # 防止数值溢出
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

class Multiclass_LogisticRegression:
    def __init__(self, n_features, n_classes=3):
        """
        初始化多类逻辑回归模型
        Args:
            n_features: 输入特征数量
            n_classes: 输出类别数量，默认为3
        """
        self.n_features = n_features
        self.n_classes = n_classes
        
        # 初始化权重：形状为 (n_classes, n_features)
        self.w = np.random.randn(self.n_classes, self.n_features) * 0.01 
        # 初始化偏置：形状为 (n_classes, 1) 或 (n_classes,)
        self.b = np.zeros((self.n_classes, 1))
    
    def forward(self, X):
        """
        前向传播
        Args:
            X: 输入数据，形状为 (batch_size, n_features)
        Returns:
            y: softmax 输出概率，形状为 (batch_size, n_classes)
        """
        # z 形状: (batch_size, n_classes)
        z = X @ self.w.T + self.b.T
        y = softmax(z)
        return y
    
    def train(self, X, Y, lr=Config.LEARNING_RATE):
        """
        训练模型（一次迭代）
        Args:
            X: 输入数据，形状为 (batch_size, n_features)
            Y: 真实标签的 one-hot 编码，形状为 (batch_size, n_classes)
            lr: 学习率
        Returns:
            loss: 交叉熵损失值
        """
        batch_size = X.shape[0]
        
        # 前向传播
        y_pred = self.forward(X)  # 形状: (batch_size, n_classes)
        
        # 计算交叉熵损失
        epsilon = 1e-15
        y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
        CCE = -np.sum(Y * np.log(y_pred_clipped)) / batch_size
        
        # 计算梯度 (softmax + 交叉熵的梯度简化为 y_pred - Y)
        # dz 形状: (batch_size, n_classes)
        dz = (y_pred - Y) / batch_size
        
        # dw 形状: (n_classes, n_features) = dz.T @ X
        dw = dz.T @ X
        
        # db 形状: (n_classes, 1) = sum(dz, axis=0, keepdims=True).T
        db = np.sum(dz, axis=0, keepdims=True).T
        
        # 更新参数
        self.w -= lr * dw
        self.b -= lr * db
        
        return CCE
    
    def predict(self, X):
        """
        预测类别
        Args:
            X: 输入数据，形状为 (batch_size, n_features)
        Returns:
            pred_labels: 预测的类别索引，形状为 (batch_size,)
        """
        y_pred = self.forward(X)  # 形状: (batch_size, n_classes)
        pred_labels = np.argmax(y_pred, axis=1)  # 获取概率最大的类别索引
        return pred_labels
    
    def predict_proba(self, X):
        """
        预测概率
        Args:
            X: 输入数据，形状为 (batch_size, n_features)
        Returns:
            y_pred: 每个类别的预测概率，形状为 (batch_size, n_classes)
        """
        return self.forward(X)