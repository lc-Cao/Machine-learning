# 测试数据导入
import sys
sys.path.append('.')

print("测试导入 make_data_multiclass...")
try:
    from data.make_data_multiclass import X_train, Y_train, Y_train_onehot, mini_batch_n, X_test, Y_test, X_train_scaled, Y_train_full
    print("✓ 成功导入所有变量")
    print(f"  X_train shape: {X_train.shape}")
    print(f"  Y_train shape: {Y_train.shape}")
    print(f"  Y_train_onehot shape: {Y_train_onehot.shape}")
    print(f"  mini_batch_n: {mini_batch_n}")
    print(f"  X_test shape: {X_test.shape}")
    print(f"  Y_test shape: {Y_test.shape}")
    print(f"  X_train_scaled shape: {X_train_scaled.shape}")
    print(f"  Y_train_full shape: {Y_train_full.shape}")
except ImportError as e:
    print(f"✗ 导入失败: {e}")
    print("\n尝试检查模块内容...")
    import data.make_data_multiclass as module
    print("模块中的变量:", [v for v in dir(module) if not v.startswith('_')])