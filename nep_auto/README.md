# NEP Auto - NEP 主动学习自动化框架

基于 MaxVol 算法的 NEP 势函数主动学习框架，提供自动化的结构选择和 DFT 标注流程。

## 📋 主要功能

本框架实现了完整的主动学习循环：

```
用户提供初始文件  →  iter_1: 第一轮迭代  →  iter_2: 第二轮...
(nep.txt, train.xyz)      ↓
                    GPUMD 探索
                         ↓
                    MaxVol 结构选择
                         ↓
                    VASP DFT 标注
                         ↓
                    NEP 重新训练
                         ↓
                    更新活跃集
                         ↓
                    (循环至收敛)
```

**核心特性**：
- ✅ **直接从 iter_1 开始** - 无需 iter_0，直接使用用户提供的初始文件
- ✅ **自动添加 DONE 标记** - 所有 job.sh 脚本自动添加用 `touch DONE`
- ✅ **智能初始化** - 自动检测并准备 iter_1，无需手动初始化
- ✅ **支持中断恢复** - 可从任意迭代继续运行
- ✅ **完整的日志记录** - 详细记录每个步骤

## 🚀 快速开始

### 1. 准备初始文件

需要提供以下文件：
- `nep.txt`: 初始 NEP 模型
- `train.xyz`: 初始训练数据（包含能量、力、应力）
- VASP 输入文件: `INCAR`, `POTCAR`, `KPOINTS`
- GPUMD 探索初始结构文件

### 2. 创建配置文件

复制并修改配置示例：

```bash
cp nep_auto/config_example.yaml my_config.yaml
# 编辑 my_config.yaml，设置文件路径和参数
```

### 3. 运行主动学习

**最简单的方式**（推荐）：

```bash
uv run nep-auto-main my_config.yaml
```

这会自动：
1. 检测 iter_1 是否存在
2. 如果不存在，自动初始化（复制初始文件、生成活跃集、准备 GPUMD）
3. 运行迭代循环直到收敛或达到最大次数

**分步运行**：

```bash
# 步骤 1: 仅初始化（可选）
uv run nep-auto-init my_config.yaml

# 步骤 2: 运行迭代
uv run nep-auto-main my_config.yaml
```

**中断后恢复**：

```bash
# 从特定迭代继续（比如程序中断了）
uv run nep-auto-main my_config.yaml --start-iter 3
```

## 📂 目录结构

运行后的工作目录结构：

```
work/
├── active_learning.log        # 日志文件
├── iter_1/                    # 第一轮迭代
│   ├──nep.txt                 # 初始/训练后的 NEP 模型
│   ├── train.xyz              # 初始/扩充后的训练数据
│   ├── active_set.asi         # 活跃集逆矩阵
│   ├── gpumd/                 # GPUMD 探索目录
│   │   ├── 300K_NVT/
│   │   │   ├── model.xyz      # 初始结构
│   │   │   ├── nep.txt        # NEP 模型
│   │   │   ├── active_set.asi # 活跃集
│   │   │   ├── run.in         # GPUMD 输入
│   │   │   ├── job.sh         # 作业脚本 (自动添加 DONE)
│   │   │   ├── DONE           # 完成标记
│   │   │   └── extrapolation_dump.xyz  # GPUMD 输出
│   │   └── 1000K_NVT/
│   │       └── ...
│   ├── large_gamma.xyz        # 合并的高 Gamma 结构
│   ├── to_add.xyz             # 选中待标注的结构
│   ├── vasp/                  # VASP DFT 计算目录
│   │   ├── task_0000/
│   │   │   ├── POSCAR
│   │   │   ├── INCAR, POTCAR, KPOINTS
│   │   │   ├── job.sh (自动添加 DONE)
│   │   │   ├── DONE
│   │   │   └── OUTCAR
│   │   └── task_0001/
│   │       └── ...
│   └── nep_train/             # NEP 训练目录
│       ├── train.xyz
│       ├── nep.in
│       ├── nep.txt
│       ├── job.sh (自动添加 DONE)
│       └── DONE
├── iter_2/                    # 第二轮迭代
│   └── ...
└── iter_3/                    # 第三轮迭代
    └── ...
```

## ⚙️ 配置文件说明

### 全局配置

```yaml
global:
  work_dir: ./work                    # 工作目录
  initial_nep_model: nep.txt          # 初始 NEP 模型
  initial_train_data: train.xyz       # 初始训练数据
  max_iterations: 10                  # 最大迭代次数
  max_structures_per_iteration: 50    # 每轮最多标注的结构数
  submit_command: "qsub job.sh"       # 作业提交命令
  check_interval: 60                  # 检查作业状态的间隔(秒)
  log_file: active_learning.log       # 日志文件
```

### VASP 配置

```yaml
vasp:
  incar_file: INCAR           # INCAR 文件路径
  potcar_file: POTCAR         # POTCAR 文件路径
  kpoints_file: KPOINTS       # KPOINTS 文件路径
  timeout: 86400              # 超时时间(秒)
  job_script: |               # 作业脚本内容
    #!/bin/bash
    #PBS -N vasp
    #PBS -l nodes=1:ppn=24
    mpirun vasp_std
    # 注意：框架会自动添加 'touch DONE'
```

### NEP 配置

```yaml
nep:
  timeout: 86400              # 超时时间(秒)
  input_content: |            # nep.in 文件内容
    version        4
    type           3 K Li Ge
    cutoff         8 4
    n_max          4 4
    basis_size     12 12
    l_max          4 2 1
    neuron         30
    batch          1000
    generation     100000
  job_script: |               # 作业脚本内容
    #!/bin/bash
    #PBS -N nep
    #PBS -l nodes=1:ppn=1:gpus=1
    nep
    # 注意：框架会自动添加 'touch DONE'
```

### GPUMD 配置

```yaml
gpumd:
  timeout: 86400              # 超时时间(秒)
  job_script: |               # 作业脚本内容
    #!/bin/bash
    #PBS -N gpumd
    gpumd
    # 注意：框架会自动添加 'touch DONE'
  conditions:
    - id: 300K_NVT           # 条件标识符
      structure_file: init.xyz
      run_in_content: |      # run.in 内容（必须包含 compute_extrapolation）
        potential        nep.txt
        velocity         300
        ensemble         nvt_ber 300 300 100
        time_step        1
        compute_extrapolation active_set.asi 0.1
        dump_exyz        100
        run              100000
```

### MaxVol 选择配置

```yaml
selection:
  gamma_tol: 1.001           # Gamma 收敛阈值
  batch_size: 10000          # 批处理大小
```

## 📦 核心模块

### 1. `config.py` - 配置加载

- 加载和验证 YAML 配置
- 自动解析相对/绝对路径
- 检查 GPUMD 配置中的 `compute_extrapolation`

### 2. `maxvol.py` - MaxVol 算法

- 描述符投影计算
- MaxVol 算法实现
- Gamma 值计算
- 活跃集生成和文件 I/O

### 3. `initialize.py` - 初始化

- 创建 iter_1 目录
- 复制初始 NEP 模型和训练数据
- 生成活跃集
- 准备第一轮 GPUMD 任务

### 4. `iteration.py` - 迭代管理

- `TaskManager`: 任务提交和监控
- `IterationManager`: 完整迭代循环
  - GPUMD 探索
  - 结构筛选（MaxVol）
  - VASP 标注
  - NEP 训练
  - 活跃集更新

### 5. `main.py` - 主程序

- 整合初始化和迭代
- 支持中断后恢复
- 异常处理和日志

## 🔄 迭代流程详解

### iter_1 (第一轮) - 特殊处理

当运行 `iter_1` 时，如果目录不存在，会自动：
1. 从配置文件读取 `initial_nep_model` 和 `initial_train_data`
2. 复制到 `iter_1/` 目录
3. 使用初始训练数据生成活跃集
4. 准备 GPUMD 探索任务

### iter_2+ (后续迭代)

每轮迭代包括 6 个步骤：

1. **GPUMD 探索**: 运行分子动力学，收集高 Gamma 结构
2. **结构筛选**: 使用 MaxVol 选择最有价值的结构
3. **VASP 标注**: 对选中结构进行 DFT 计算
4. **扩充数据集**: 将 DFT 结果追加到训练集
5. **NEP 训练**: 用更新的数据集重新训练模型
6. **更新活跃集**: 重新计算活跃集
7. **准备下一轮**: 为下一轮 GPUMD 准备文件

## ⚠️ 注意事项

### 自动添加 DONE 标记

**重要**：框架会自动在所有 `job.sh` 末尾添加 `touch DONE`。您的作业脚本中**不需要**手动添加此行。

示例配置中的作业脚本：

```yaml
job_script: |
  #!/bin/bash
  #PBS -N myjob
  #PBS -l nodes=1:ppn=24
  
  cd $PBS_O_WORKDIR
  mpirun my_program
  
  # 不需要写 'touch DONE'，框架会自动添加！
```

### 路径解析规则

- **绝对路径**: 直接使用
- **相对路径**: 基于 `work_dir` 解析

### GPUMD 要求

每个 `run_in_content` **必须包含** `compute_extrapolation` 指令，例如：

```
compute_extrapolation active_set.asi 0.1
```

否则无法收集高 Gamma 结构。

### 作业调度系统

框架支持任意作业调度系统（PBS, SLURM, 本地运行等），只需在 `submit_command` 中指定提交命令。

示例：
- PBS: `qsub job.sh`
- SLURM: `sbatch job.sh`
- 本地: `bash job.sh &`

## 🛠️ 依赖项

```bash
uv add numpy scipy pyyaml ase tqdm
uv add "git+https://github.com/bigd4/PyNEP.git"
```

## 📝 示例使用流程

```bash
# 1. 安装依赖
uv sync

# 2. 准备配置文件
cp nep_auto/config_example.yaml my_config.yaml
# 编辑 my_config.yaml

# 3. 验证配置（可选）
uv run nep-auto-config my_config.yaml

# 4. 运行主动学习
uv run nep-auto-main my_config.yaml

# 5. 监控日志
tail -f work/active_learning.log
```

## 🐛 故障排除

### 问题 1: "上一轮目录不存在"

**原因**: 尝试从 iter_N 开始，但 iter_{N-1} 不存在

**解决**: 从 iter_1 开始：
```bash
uv run nep-auto-main my_config.yaml --start-iter 1
```

### 问题 2: "NEP 模型元素类型不匹配"

**原因**: nep.txt 中的元素与 train.xyz 不一致

**解决**: 检查 nep.txt 第一行的元素列表，确保与训练数据匹配

### 问题 3: 作业一直不完成

**原因**: DONE 文件未创建

**解决**: 
- 检查作业是否真的完成
- 确认作业脚本中的 `touch DONE` 被执行（框架会自动添加）
- 检查作业目录权限

## 📚 更多文档

- `OVERVIEW.md`: 开发者文档和模块详解
- `config_example.yaml`: 配置文件示例

---

## 更新日志

### v2.0 - 2024-12-25

- ✅ **移除 iter_0**：直接从 iter_1 开始，简化流程
- ✅ **自动添加 DONE**：所有 job.sh 自动添加 `touch DONE`
- ✅ **智能初始化**：iter_1 自动从用户提供的初始文件获取
- ✅ **改进的中断恢复**：更好的流程控制和错误提示

基于 nep_maker 项目的 CPU 版本重构，整合了所有 MaxVol 相关功能。
