import numpy as np
from config import Config

def softmax(x):
    """
    计算 softmax 函数
    输入: x 形状为 (batch_size, n_classes)
    输出: 每个样本在每个类别上的概率，形状与输入相同
    """
    x = np.clip(x, -500, 500)
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


class Multiclass_LogisticRegression_Adam:
    """
    带 Adam 优化器 + L1/L2 正则化的多类逻辑回归模型
    """
    def __init__(self, n_features, n_classes=3,
                 reg_type='none', reg_lambda=0.0,
                 lr=Config.LEARNING_RATE,
                 beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        Args:
            n_features: 输入特征数量
            n_classes: 输出类别数量
            reg_type: 正则化类型 ('none' / 'l1' / 'l2')
            reg_lambda: 正则化系数 λ
            lr: Adam 学习率 (通常比 SGD 小，建议 0.001)
            beta1: Adam 一阶矩衰减系数
            beta2: Adam 二阶矩衰减系数
            epsilon: 数值稳定项
        """
        self.n_features = n_features
        self.n_classes = n_classes
        self.reg_type = reg_type
        self.reg_lambda = reg_lambda
        self.lr = lr

        # Adam 参数
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0  # 时间步计数器

        # 权重和偏置
        self.w = np.random.randn(self.n_classes, self.n_features) * 0.01
        self.b = np.zeros((self.n_classes, 1))

        # Adam 一阶矩（动量）和二阶矩（RMSProp 缓存）
        self.m_w = np.zeros_like(self.w)
        self.v_w = np.zeros_like(self.w)
        self.m_b = np.zeros_like(self.b)
        self.v_b = np.zeros_like(self.b)

    def forward(self, X):
        """前向传播: X(batch, n_features) → probs(batch, n_classes)"""
        z = X @ self.w.T + self.b.T
        return softmax(z)

    def train(self, X, Y, lr=None):
        """
        单次 Mini-batch Adam 训练
        Args:
            X: (batch_size, n_features)
            Y: one-hot 标签 (batch_size, n_classes)
            lr: 学习率（可选覆盖，通常用 __init__ 中的值）
        Returns:
            total_loss: 交叉熵 + 正则化项
        """
        if lr is not None:
            self.lr = lr  # 允许外部动态调整学习率

        batch_size = X.shape[0]
        self.t += 1  # 时间步递增

        # 前向传播
        y_pred = self.forward(X)

        # 交叉熵损失
        epsilon = 1e-15
        y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
        CCE = -np.sum(Y * np.log(y_pred_clipped)) / batch_size

        # 梯度计算
        dz = (y_pred - Y) / batch_size          # (batch_size, n_classes)
        dw = dz.T @ X                            # (n_classes, n_features)
        db = np.sum(dz, axis=0, keepdims=True).T # (n_classes, 1)

        # ---- 正则化 ----
        reg_loss = 0.0
        if self.reg_type == 'l2':
            reg_loss = 0.5 * self.reg_lambda * np.sum(self.w ** 2)
            dw += self.reg_lambda * self.w
        elif self.reg_type == 'l1':
            reg_loss = self.reg_lambda * np.sum(np.abs(self.w))
            dw += self.reg_lambda * np.sign(self.w)

        # ---- Adam 更新 ----
        # 一阶矩估计
        self.m_w = self.beta1 * self.m_w + (1 - self.beta1) * dw
        self.m_b = self.beta1 * self.m_b + (1 - self.beta1) * db

        # 二阶矩估计
        self.v_w = self.beta2 * self.v_w + (1 - self.beta2) * (dw ** 2)
        self.v_b = self.beta2 * self.v_b + (1 - self.beta2) * (db ** 2)

        # 偏差校正
        m_w_hat = self.m_w / (1 - self.beta1 ** self.t)
        v_w_hat = self.v_w / (1 - self.beta2 ** self.t)
        m_b_hat = self.m_b / (1 - self.beta1 ** self.t)
        v_b_hat = self.v_b / (1 - self.beta2 ** self.t)

        # 参数更新
        self.w -= self.lr * m_w_hat / (np.sqrt(v_w_hat) + self.epsilon)
        self.b -= self.lr * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)

        return CCE + reg_loss

    def predict(self, X):
        """预测类别"""
        y_pred = self.forward(X)
        return np.argmax(y_pred, axis=1)

    def predict_proba(self, X):
        """预测概率"""
        return self.forward(X)