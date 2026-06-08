import numpy as np
from config import Config
import copy

def softmax(x):
    x = np.clip(x, -500, 500)
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


class Multiclass_LogisticRegression_EarlyStop:
    """
    带早停机制 + L1/L2 正则化的多类逻辑回归模型
    """
    def __init__(self, n_features, n_classes=3,
                 reg_type='none', reg_lambda=0.0,
                 patience=5):
        """
        Args:
            n_features: 输入特征数量
            n_classes: 输出类别数量
            reg_type: 正则化类型 ('none' / 'l1' / 'l2')
            reg_lambda: 正则化系数 λ
            patience: 早停耐心值（连续不提升轮数）
        """
        self.n_features = n_features
        self.n_classes = n_classes
        self.reg_type = reg_type
        self.reg_lambda = reg_lambda
        self.patience = patience

        self.w = np.random.randn(self.n_classes, self.n_features) * 0.01
        self.b = np.zeros((self.n_classes, 1))

        # 早停状态
        self.best_val_loss = float('inf')
        self.best_w = None
        self.best_b = None
        self.wait = 0
        self.stopped_epoch = -1
        self.should_stop = False

    def forward(self, X):
        z = X @ self.w.T + self.b.T
        return softmax(z)

    def compute_loss(self, X, Y):
        """计算损失（含正则化项），不更新参数"""
        batch_size = X.shape[0]
        y_pred = self.forward(X)
        epsilon = 1e-15
        y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
        CCE = -np.sum(Y * np.log(y_pred_clipped)) / batch_size

        reg_loss = 0.0
        if self.reg_type == 'l2':
            reg_loss = 0.5 * self.reg_lambda * np.sum(self.w ** 2)
        elif self.reg_type == 'l1':
            reg_loss = self.reg_lambda * np.sum(np.abs(self.w))

        return CCE + reg_loss

    def train_step(self, X, Y, lr=Config.LEARNING_RATE):
        """单步参数更新，返回训练损失"""
        batch_size = X.shape[0]
        y_pred = self.forward(X)

        epsilon = 1e-15
        y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
        CCE = -np.sum(Y * np.log(y_pred_clipped)) / batch_size

        dz = (y_pred - Y) / batch_size
        dw = dz.T @ X
        db = np.sum(dz, axis=0, keepdims=True).T

        reg_loss = 0.0
        if self.reg_type == 'l2':
            reg_loss = 0.5 * self.reg_lambda * np.sum(self.w ** 2)
            dw += self.reg_lambda * self.w
        elif self.reg_type == 'l1':
            reg_loss = self.reg_lambda * np.sum(np.abs(self.w))
            dw += self.reg_lambda * np.sign(self.w)

        self.w -= lr * dw
        self.b -= lr * db

        return CCE + reg_loss

    def validate(self, X_val, Y_val, Y_val_onehot):
        """
        验证并执行早停检查
        Returns:
            val_loss: 验证集损失
            improved: 本轮是否提升
        """
        val_loss = self.compute_loss(X_val, Y_val_onehot)

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_w = self.w.copy()
            self.best_b = self.b.copy()
            self.wait = 0
            return val_loss, True
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.should_stop = True
                self.stopped_epoch = getattr(self, '_current_epoch', -1)
                # 回滚到最佳权重
                self.w = self.best_w.copy()
                self.b = self.best_b.copy()
            return val_loss, False

    def restore_best(self):
        """恢复到最佳权重"""
        if self.best_w is not None:
            self.w = self.best_w.copy()
            self.b = self.best_b.copy()

    def predict(self, X):
        y_pred = self.forward(X)
        return np.argmax(y_pred, axis=1)

    def predict_proba(self, X):
        return self.forward(X)