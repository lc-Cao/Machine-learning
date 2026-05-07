import numpy as np
from config import Config

def sigmoid(x):
    # 使用 np.clip 防止数值溢出（针对真实高维数据）
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

class Logistic_Regression:
    def __init__(self, size):
        # 这里的 size 应该对应真实数据的特征数（例如乳腺癌数据集的 30）
        self.size = size 
        # 初始化权重：建议使用较小的随机值，有助于收敛
        self.w = np.random.randn(1, self.size) * 0.01 
        self.b = 0.0
    
    # 前向传播
    def forward(self, X): 
        # X 形状: (batch_size, 30)
        # self.w.T 形状: (30, 1)
        z = X @ self.w.T + self.b 
        y = sigmoid(z)
        return y
    
    # 训练模型
    def train(self, X, Y, lr=Config.LEARNING_RATE):
        batch_size = X.shape[0]
        y = self.forward(X)
        
        # 1. 交叉熵损失 (增加 epsilon 防止 log(0) 报错)
        epsilon = 1e-15
        CCE = -np.sum(Y * np.log(y + epsilon) + (1 - Y) * np.log(1 - y + epsilon)) / batch_size 
        
        # 2. 计算梯度 (核心优化：Sigmoid + CrossEntropy 的导数简化为 y - Y)
        # dz 形状: (batch_size, 1)
        dz = (y - Y) / batch_size 
        
        # dw 形状: (1, 30) -> (1, batch_size) @ (batch_size, 30)
        dw = dz.T @ X 
        
        # db 形状: 标量
        db = np.sum(dz)
        
        # 3. 更新参数
        self.w -= lr * dw
        self.b -= lr * db
        
        return CCE
    
    # 预测种类
    def predict(self, X):
        y = self.forward(X)
        return (y >= 0.5).astype(int)