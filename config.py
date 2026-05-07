class Config:
    SEED = 2026 # 随机种子

    # 数据参数
    N_SAMPLES = 1280        # 数据数量
    N_FEATURES = 54         # 数据特征个数 (covtype数据集)
    TEST_RATE = 0.2         # 测试集占比

    # 训练参数
    BATCH_SIZE = 128        # BATCH的大小 (适合大规模数据)
    EPOCH = 150             # 训练轮次 (增加到150轮，因为样本更多)
    LEARNING_RATE = 0.01    # 学习率
    PRINT_SIZE = 15         # 损失曲线的点数 (每10轮打印一次)
