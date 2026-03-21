import numpy as np
from config import Config

np.random.seed(Config.SEED)   

"""
数据来源：使用 sklearn 的 make_classification 生成合成数据
- 样本数:Config.N_SAMPLES
- 特征数:Config.N_FEATURES
- 类别数:2
- 随机种子:Config.SEED

数据划分：使用 train_test_split 划分
- 训练集:80%  （可以更改）
- 测试集:20%  （可以更改）

数据分块：
- 分块数据,用于训练自己的模型
-- 数据块大小: Config.BATCH_SIZE
-- 数据块数量: Config.N_SAMPLES / Config.BATCH_SIZE
- 完整数据,用于训练sk模型
"""

# 数据创建
from sklearn.datasets import make_classification
X,Y = make_classification(n_samples = Config.N_SAMPLES,                         # 数据总数量
                          n_features = Config.N_FEATURES,                       # 每个数据的特征数量
                          n_classes = 2,                                        # 数据的种类
                          n_informative=Config.N_FEATURES,                      # 有效特征数
                          n_redundant=0,                                        # 冗余特征数
                          n_repeated=0,                                         # 重复特征数
                          random_state = Config.SEED                            # 随机种子
                        )

# 数据划分
from sklearn.model_selection import train_test_split
X_train_full,X_test,Y_train_full,Y_test = train_test_split(X,                   # 数据 X，与 X_train_2d, X_test 对应
                                                       Y,                       # 数据 Y，与 Y_train_2d, Y_test 对应
                                                 test_size = Config.TEST_RATE,  # 测试数据占比
                                                 random_state = Config.SEED     # 随机种子
                                                )

# 训练数据分块
mini_batch_size = Config.BATCH_SIZE                                             # 每个数据块的大小                                                
mini_batch_n = len(X_train_full) // mini_batch_size                             # 数据块的数量,多余的数据丢掉

X_train = X_train_full.reshape(mini_batch_n,mini_batch_size,Config.N_FEATURES)  # 把 X_train_full 按照数据块整理 
Y_train = Y_train_full.reshape(-1,1).reshape(mini_batch_n,mini_batch_size,1)    # 把 Y_train_full 按照数据块整理
