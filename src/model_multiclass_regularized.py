import numpy as np
from config import Config

def softmax(x):
    """
    计算 softmax 函数
    输入: x 形状为 (batch_size, n_classes)
    输出: 每个样本在每个类别上的概率，形状与输入相同
    """
    x = np.clip(x, -500, 500)  # 防止数值溢出
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


class Multiclass_LogisticRegression_Regularized:
    """
    带 L1/L2 正则化的多类逻辑回归模型
    """
    def __init__(self, n_features, n_classes=3, reg_type='l2', reg_lambda=0.01):
        """
        初始化模型
        Args:
            n_features: 输入特征数量
            n_classes: 输出类别数量
            reg_type: 正则化类型 ('none' / 'l1' / 'l2')
            reg_lambda: 正则化系数 λ
        """
        self.n_features = n_features
        self.n_classes = n_classes
        self.reg_type = reg_type         # 'none', 'l1', 'l2'
        self.reg_lambda = reg_lambda     # 正则化系数
        
        self.w = np.random.randn(self.n_classes, self.n_features) * 0.01
        self.b = np.zeros((self.n_classes, 1))

    def forward(self, X):
        """
        前向传播
        X: (batch_size, n_features) → y: (batch_size, n_classes)
        """
        z = X @ self.w.T + self.b.T
        return softmax(z)

    def train(self, X, Y, lr=Config.LEARNING_RATE):
        """
        单次 Mini-batch 训练，支持 L1/L2 正则化
        Args:
            X: (batch_size, n_features)
            Y: one-hot 标签 (batch_size, n_classes)
            lr: 学习率
        Returns:
            total_loss: 交叉熵 + 正则化项
        """
        batch_size = X.shape[0]

        # 前向传播
        y_pred = self.forward(X)  # (batch_size, n_classes)

        # 交叉熵损失
        epsilon = 1e-15
        y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
        CCE = -np.sum(Y * np.log(y_pred_clipped)) / batch_size

        # 梯度（softmax + 交叉熵简化）
        dz = (y_pred - Y) / batch_size           # (batch_size, n_classes)
        dw = dz.T @ X                             # (n_classes, n_features)
        db = np.sum(dz, axis=0, keepdims=True).T  # (n_classes, 1)

        # ---- 正则化 ----
        reg_loss = 0.0
        if self.reg_type == 'l2':
            # L2: (λ/2) * Σw², 梯度: λ * w
            reg_loss = 0.5 * self.reg_lambda * np.sum(self.w ** 2)
            dw += self.reg_lambda * self.w
        elif self.reg_type == 'l1':
            # L1: λ * Σ|w|, 梯度: λ * sign(w)
            reg_loss = self.reg_lambda * np.sum(np.abs(self.w))
            dw += self.reg_lambda * np.sign(self.w)

        # 参数更新
        self.w -= lr * dw
        self.b -= lr * db

        return CCE + reg_loss

    def predict(self, X):
        """
        预测类别
        X: (batch_size, n_features) → labels: (batch_size,)
        """
        y_pred = self.forward(X)
        return np.argmax(y_pred, axis=1)

    def predict_proba(self, X):
        """
        预测概率
        X: (batch_size, n_features) → probs: (batch_size, n_classes)
        """
        return self.forward(X)