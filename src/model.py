import numpy as np
from config import Config

def sigmoid(x) :
    return 1 / ( 1 + np.exp(-x) )

def sigmoid_derivative(y) :
    return y * ( 1 - y )

class Logistic_Regression :
    def __init__(self,size) :
        self.size = size                                            # size 大小为数据的特征数量(N_features)

        self.w = np.random.uniform(-1,1,size = (1,self.size))       # self.w : 1 * size   
        self.b = np.random.uniform(-1,1)                            # self.b : 标量
    
    # 前向传播
    def forward(self,X) :                                           # X : X_size * size
        z = X @ self.w.T                                            # z : X_size * 1
        h = z + self.b                                              # h : X_size * 1
        y = sigmoid(h)                                              # y : X_size * 1
        return y
    
    # 反向传播
    def train(self,X,Y,lr = Config.LEARNING_RATE) :                 # Y : X_size * 1
        y = self.forward(X)
        batch_size = len(X)                                         # batch_size = X_size
        # 交叉熵损失
        CCE = -np.sum(Y * np.log(y) + (1 - Y) * np.log(1 - y)) / batch_size 
        # 计算梯度
        dy = ((y - Y) / ( y * ( 1- y ) )) / batch_size              # dy : X_size * 1
        dh = dy * sigmoid_derivative(y)                             # dh : X_size * 1
        dz = ( y - Y ) / batch_size                                 # dz = dh  
        dw = dz.T @ X                                               # dw : 1 * size
        db = np.sum(dh)                                             # 标量
        # 更新参数
        self.w -= lr * dw
        self.b -= lr * db
        return CCE
    
    # 预测种类
    def predict(self,X) :
        y = self.forward(X)
        y = (y >= 0.5).astype(int)

        return y