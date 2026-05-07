```markdown
# 逻辑回归从零实现

## 项目简介
使用 NumPy 从零实现逻辑回归，Mini-batch 梯度下降训练，与 sklearn 对比。

## 安装与运行

### 1. 克隆项目
```bash
git clone https://github.com/liqingchao456789-ui/ML-4_project
cd ML-4_project
```

### 2. 创建虚拟环境
```bash
python3 -m venv .venv      #Windows输入python -m venv .venv
source .venv/bin/activate  # Mac/Linux
# 或
.venv\Scripts\activate     # Windows
```

### 3. 安装依赖
```bash
pip install -r requirements.txt
```

### 4. 运行
```bash
python main.py
```

## 项目结构
```
ML-4_project/
├── main.py              # 主程序
├── config.py            # 配置文件
├── requirements.txt     # 依赖包
├── data/
│   └── make_data.py     # 数据生成
├── src/
│   ├── model.py         # 逻辑回归模型
│   └── utils.py         # 工具函数
├── models/              # 模型保存（运行生成）
└── results/             # 结果保存（运行生成）
```

## 作者
李庆超 
