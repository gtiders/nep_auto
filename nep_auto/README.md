# NEP Auto - NEP 主动学习自动化框架

这是一个基于 MaxVol 算法的 NEP 势函数主动学习框架，提供自动化的结构选择和 DFT 标注流程。

## 目录结构

```
nep_auto/
├── __init__.py           # 模块初始化
├── config.py            # 配置加载模块
├── config_example.yaml  # 配置文件示例
├── initialize.py        # 初始化脚本（第 0 步）
├── maxvol.py           # MaxVol 算法核心模块
└── README.md           # 本文档
```

## 核心模块说明

### 1. `config.py` - 配置加载模块

负责从 YAML 文件加载所有配置参数，包括：
- **全局配置**: 工作目录、迭代次数、日志等
- **VASP 配置**: DFT 计算的输入文件路径和作业脚本
- **NEP 配置**: NEP 训练参数和作业脚本
- **GPUMD 配置**: 分子动力学探索条件
- **MaxVol 配置**: 算法参数

**主要功能**:
- 自动解析相对路径（以 `work_dir` 为基准）
- 验证文件是否存在
- 检测 GPUMD run.in 是否包含必需的 `compute_extrapolation` 指令

**使用示例**:
```python
from nep_auto.config import load_config, print_config_summary

config = load_config("config.yaml")
print_config_summary(config)
```

### 2. `maxvol.py` - MaxVol 算法模块

提供基于 MaxVol（最大体积）算法的主动学习结构选择功能。

**核心功能**:
- **描述符投影计算** (`compute_descriptor_projection`): 计算 NEP 描述符
- **MaxVol 算法** (`compute_maxvol`): 选择最具代表性的结构子集
- **Gamma 值计算** (`compute_gamma`): 评估外推程度
- **活跃集生成** (`generate_active_set`): 从训练集生成活跃集
- **ASI 文件 I/O** (`write_asi_file`, `read_asi_file`): 保存/读取活跃集逆矩阵

**高级选择函数**:
```python
from nep_auto.maxvol import (
    select_active_set,           # 从训练集选择活跃集
    select_extension_structures, # 从候选集选择待标注结构
    filter_high_gamma_structures # 筛选高 Gamma 结构
)
```

### 3. `initialize.py` - 初始化脚本

这是独立于普通迭代的**第 0 步**，负责：

1. **创建工作目录结构** (`iter_0/`)
2. **复制初始文件** (nep.txt, train.xyz)
3. **生成第一个活跃集** (active_set.asi)
4. **准备 GPUMD 探索任务**
   - 为每个探索条件创建独立目录
   - 复制必要的文件（nep.txt, active_set.asi, 结构文件）
   - 生成 run.in 和作业脚本

**使用方法**:
```bash
python -m nep_auto.initialize config.yaml
```

### 4. `iteration.py` - 迭代管理模块

管理主动学习的核心迭代循环。

**TaskManager（任务管理器）**:
- `submit_job()`: 在指定目录提交作业
- `wait_for_completion()`: 等待作业完成（检测 DONE 文件）

**IterationManager（迭代管理器）**:
- `run_gpumd()`: 运行 GPUMD 探索，收集高 Gamma 结构
- `select_structures()`: 使用 MaxVol 选择待标注结构
- `run_vasp()`: 提交 VASP DFT 计算并收集结果
- `run_nep()`: 训练 NEP 模型
- `update_active_set()`: 更新活跃集
- `prepare_next_gpumd()`: 准备下一轮 GPUMD 探索
- `run_iteration()`: 运行一次完整迭代

**单次迭代流程**:
```python
from nep_auto import IterationManager, load_config, setup_logger

config = load_config("config.yaml")
logger = setup_logger(config.global_config.log_file)
manager = IterationManager(config, logger)

# 运行迭代 1
should_continue = manager.run_iteration(iter_num=1)
```

### 5. `main.py` - 主程序入口

整合初始化和迭代流程的完整主动学习程序。

**主要功能**:
- 加载配置并验证
- 执行初始化（如果从头开始）
- 运行迭代循环直到收敛或达到最大次数
- 支持中断后从指定迭代继续
- 异常处理和日志记录

**使用方法**:
```bash
# 从头开始运行完整流程
python -m nep_auto.main config.yaml

# 从指定迭代继续（比如程序中断后）
python -m nep_auto.main config.yaml --start-iter 5
```

## 快速开始

### 步骤 1: 准备配置文件

复制并修改 `config_example.yaml`:

```bash
cp config_example.yaml my_config.yaml
```

**必须提供的初始文件**:
- `nep.txt`: 初始 NEP 模型（可以是预训练的或随机初始化的）
- `train.xyz`: 初始训练数据（必须包含能量、力、应力信息）
- VASP 输入文件: `INCAR`, `POTCAR`, `KPOINTS`
- GPUMD 初始结构: 每个探索条件的初始结构文件

### 步骤 2: 验证配置

```bash
python -m nep_auto.config my_config.yaml
```

这会打印配置摘要并检查所有文件是否存在。

### 步骤 3: 执行初始化

```bash
python -m nep_auto.initialize my_config.yaml
```

初始化完成后，工作目录结构如下:

```
work/
├── active_learning.log          # 日志文件
└── iter_0/
    ├── nep.txt                  # NEP 模型
    ├── train.xyz                # 训练数据
    ├── active_set.asi           # 活跃集逆矩阵
    ├── active_set.xyz           # 活跃集结构（可选）
    ├── DONE                     # 完成标记
    └── gpumd/
        ├── 300K_NVT/
        │   ├── model.xyz
        │   ├── nep.txt
        │   ├── active_set.asi
        │   ├── run.in
        │   └── job.sh
        ├── 1000K_NVT/
        │   └── ...
        └── NPT_500K/
            └── ...
```

### 步骤 4: 运行主动学习流程

**选项 A: 完全自动化运行**

```bash
# 从头开始运行完整的主动学习循环
python -m nep_auto.main my_config.yaml
```

这将自动执行：
1. 初始化（迭代 0）
2. 迭代 1, 2, 3, ... 直到收敛或达到最大迭代次数

**选项 B: 分步骤运行**

```bash
# 1. 仅初始化
python -m nep_auto.initialize my_config.yaml

# 2. 手动提交初始 GPUMD 任务
cd work/iter_0/gpumd/300K_NVT && qsub job.sh
cd work/iter_0/gpumd/1000K_NVT && qsub job.sh
# ... 为每个条件提交

# 3. 等待完成后，从迭代 1 继续
python -m nep_auto.main my_config.yaml --start-iter 1
```

**选项 C: 程序中断后继续**

```bash
# 如果程序在迭代 5 中断，从迭代 5 继续
python -m nep_auto.main my_config.yaml --start-iter 5
```

### 步骤 5: 监控进度

查看日志文件：
```bash
tail -f work/active_learning.log
```

检查工作目录结构：
```

## 配置文件关键参数说明

### 全局配置

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `work_dir` | 工作目录（所有相对路径的基准） | `./work` |
| `max_iterations` | 最大迭代次数 | 100 |
| `max_structures_per_iteration` | 每轮最多标注的结构数 | 50 |
| `initial_nep_model` | 初始 NEP 模型路径 | `nep.txt` |
| `initial_train_data` | 初始训练数据路径 | `train.xyz` |
| `submit_command` | 任务提交命令 | `qsub job.sh` |

### VASP 配置

所有路径都指向**文件路径**（不是内容字符串）:
- `incar_file`: INCAR 文件路径
- `potcar_file`: POTCAR 文件路径
- `kpoints_file`: KPOINTS 文件路径
- `job_script`: 作业脚本**内容**（多行字符串）

### NEP 配置

- `input_content`: nep.in 文件**内容**（多行字符串，直接作为输入）
- `job_script`: 作业脚本内容

### GPUMD 配置

每个 `condition` 必须包含:
- `id`: 唯一标识符
- `structure_file`: 初始结构文件路径
- `run_in_content`: run.in 文件内容（**必须包含 `compute_extrapolation` 指令**）

### MaxVol 配置

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `gamma_tol` | 收敛阈值（gamma < gamma_tol 时停止） | 1.001 |
| `batch_size` | 批处理大小（避免内存溢出） | 10000 |

## 主动学习流程概述

```
第 0 步 (初始化):
    ├── 复制 nep.txt 和 train.xyz
    ├── 生成活跃集 (select_active_set)
    └── 准备 GPUMD 任务

第 1+ 步 (普通迭代):
    ├── 提交并等待 GPUMD 完成
    ├── 收集高 Gamma 结构 (large_gamma.xyz)
    ├── 选择待标注结构 (select_extension_structures)
    ├── 限制数量 (max_structures_per_iteration)
    ├── VASP DFT 计算
    ├── 追加到训练集
    ├── 重新训练 NEP
    ├── 更新活跃集
    └── 继续下一轮...
```

## 依赖项

- Python >= 3.9
- numpy
- scipy
- pyyaml
- ase
- pynep (https://github.com/bigd4/PyNEP)

安装依赖:
```bash
uv add pyyaml
uv add "git+https://github.com/bigd4/PyNEP.git"
```

## 注意事项

1. **初始文件要求**:
   - `train.xyz` 必须包含能量、力、应力信息
   - `nep.txt` 必须与训练数据的元素类型匹配

2. **路径解析规则**:
   - **相对路径**: 基于 `work_dir` 解析
   - **绝对路径**: 直接使用

3. **GPUMD 配置**:
   - 每个 `run_in_content` **必须包含** `compute_extrapolation` 指令
   - 否则程序会在加载配置时报错

4. **任务状态检测**:
   - 通过检测 `DONE` 文件判断任务是否完成
   - 提交任务时会自动切换到对应目录

## 开发者信息

基于 `nep_maker` 项目的 CPU 版本重构，整合了所有 MaxVol 相关功能到单一模块。

主要改进:
- 统一的配置管理
- 规范的函数签名和类型注解
- 中文文档字符串
- 移除 GPU 依赖，仅保留 CPU 版本
- 模块化设计，便于扩展
