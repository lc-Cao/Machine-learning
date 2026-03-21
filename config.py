class Config:
    SEED = 2026 # 随机种子

    # 数据参数
    N_SAMPLES = 1280        # 数据数量
    N_FEATURES = 8          # 数据特征个数
    TEST_RATE = 0.2         # 测试集占比

    # 训练参数
    BATCH_SIZE = 32         # BATCH的大小
    EPOCH = 100             # 训练轮次
    LEARNING_RATE = 0.01    # 学习率
    PRINT_SIZE = 100        # 损失曲线的点数