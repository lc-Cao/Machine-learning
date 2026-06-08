# 逻辑回归从零实现

## 项目简介
使用 NumPy 从零实现逻辑回归（二分类 + 三分类），Mini-batch 梯度下降训练，与 sklearn 对比。支持 3 个数据集：Breast Cancer、Digits、Covtype。

## 项目结构
```
ML-4_project/
├── main.py                        # 二分类主程序 (Covtype)
├── main_breast_cancer.py          # 二分类主程序 (Breast Cancer)
├── main_digits.py                 # 二分类主程序 (Digits 0/1)
├── main_multiclass.py             # 三分类主程序 (Covtype 3类)
├── config.py                      # 配置文件
├── requirements.txt               # 依赖包
├── data/
│   ├── make_data.py               # 二分类数据 (Covtype)
│   ├── make_data_breast_cancer.py # 二分类数据 (Breast Cancer)
│   ├── make_data_digits.py        # 二分类数据 (Digits 0/1)
│   └── make_data_multiclass.py    # 三分类数据 (Covtype 3类)
├── src/
│   ├── model.py                   # 二分类逻辑回归模型 (Sigmoid)
│   ├── model_multiclass.py        # 多分类逻辑回归模型 (Softmax)
│   ├── utils.py                   # 二分类工具函数 (评估/可视化/保存)
│   └── utils_multiclass.py        # 多分类工具函数
├── docs/                          # 实验报告
├── scripts/                       # 辅助脚本
├── models/                        # 模型保存（运行生成）
└── results/                       # 结果保存（运行生成）
```

## 数据集

| 数据集 | 特征数 | 类别 | 样本数 | 主程序 |
|--------|--------|------|--------|--------|
| Breast Cancer | 30 | 2 (恶性/良性) | 569 | `main_breast_cancer.py` |
| Digits | 64 | 2 (0/1) | ~360 | `main_digits.py` |
| Covtype | 54 | 2 (类别1/2) | 50,000 | `main.py` |
| Covtype 多分类 | 54 | 3 (类别1/2/3) | 45,000 | `main_multiclass.py` |

## 安装与运行

### 1. 创建虚拟环境
```bash
python -m venv .venv
.venv\Scripts\activate     # Windows
# source .venv/bin/activate  # Mac/Linux
```

### 2. 安装依赖
```bash
pip install -r requirements.txt
```

### 3. 运行

**Breast Cancer 二分类：**
```bash
python main_breast_cancer.py
```

**Digits 二分类：**
```bash
python main_digits.py
```

**Covtype 二分类：**
```bash
python main.py
```

**Covtype 三分类：**
```bash
python main_multiclass.py
```

## 中期报告结果快照

| 模型/设置 | 准确率 | 备注 |
|-----------|--------|------|
| 自实现二分类 (Breast Cancer) | 100.00% | 优于 sklearn (99.12%) |
| 自实现二分类 (Digits 0/1) | 100.00% | 与 sklearn 持平 |
| 自实现二分类 (Covtype 50k) | 77.66% | 小幅超越 sklearn (77.64%) |
| 自实现三分类 (Covtype 45k) | 77.39% | 与 sklearn (77.66%) 仅差 0.27% |

## 下阶段计划
- [x] 正则化优化 (L1/L2)
- [x] Adam 优化器
- [x] 早停机制
- [ ] Covtype 全 7 类多分类

## 作者
张振 24307110014 / 曹乐宸 24300190047 / 李庆超 24307110081