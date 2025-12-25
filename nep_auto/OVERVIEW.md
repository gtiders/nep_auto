# NEP Auto - 项目概览

## 项目结构

```
nep_auto/
├── __init__.py              # 模块初始化，导出主要接口
├── config.py               # 配置加载模块
├── config_example.yaml     # 配置文件示例
├── initialize.py           # 初始化脚本（迭代 0）
├── iteration.py            # 迭代管理模块（迭代 1+）
├── main.py                 # 主程序入口
├── maxvol.py              # MaxVol 算法核心模块
├── README.md              # 用户文档
└── OVERVIEW.md            # 本文档（开发者文档）
```

## 模块依赖关系

```
main.py
  ├── config.py (加载配置)
  ├── initialize.py (初始化)
  │   ├── config.py
  │   └── maxvol.py (select_active_set, write_asi_file)
  └── iteration.py (迭代循环)
      ├── config.py
      └── maxvol.py (select_extension_structures, select_active_set)

maxvol.py
  └── pynep (NEP 计算)
```

## 数据流

### 初始化阶段（迭代 0）

```
输入:
  - initial_nep_model (nep.txt)
  - initial_train_data (train.xyz)
  - VASP 输入文件 (INCAR, POTCAR, KPOINTS)
  - GPUMD 初始结构

处理流程:
  1. 复制 nep.txt, train.xyz → iter_0/
  2. 计算描述符投影 (compute_descriptor_projection)
  3. 生成活跃集 (generate_active_set)
  4. 保存 active_set.asi
  5. 准备 GPUMD 任务目录

输出:
  - iter_0/nep.txt
  - iter_0/train.xyz
  - iter_0/active_set.asi
  - iter_0/gpumd/<condition_id>/
```

### 迭代阶段（迭代 1+）

#### 步骤 1: GPUMD 探索

```
输入:
  - iter_N/gpumd/<condition_id>/
    ├── nep.txt
    ├── active_set.asi
    ├── model.xyz
    └── run.in (包含 compute_extrapolation)

GPUMD 运行:
  - 每 check_interval 步计算 Gamma
  - 当 gamma_low < max_gamma < gamma_high 时
  - 输出结构到 extrapolation_dump.xyz

输出:
  - iter_N/gpumd/<condition_id>/extrapolation_dump.xyz
  - iter_N/large_gamma.xyz (合并所有条件)
```

#### 步骤 2: 结构筛选

```
输入:
  - iter_N/train.xyz (当前训练集)
  - iter_N/large_gamma.xyz (候选结构)
  - iter_N/nep.txt (当前模型)

处理:
  1. 合并 train + candidates
  2. 计算描述符投影
  3. 执行 MaxVol 算法
  4. 筛选仅来自 candidates 的新结构
  5. 限制数量 (max_structures_per_iteration)

输出:
  - iter_N/to_add.xyz (待标注结构)
```

#### 步骤 3: VASP DFT 标注

```
输入:
  - iter_N/to_add.xyz

处理:
  1. 为每个结构创建 VASP 计算目录
  2. 复制 INCAR, POTCAR, KPOINTS
  3. 生成 POSCAR
  4. 提交作业
  5. 等待完成
  6. 从 OUTCAR 读取能量、力、应力

输出:
  - iter_N/vasp/task_XXXX/OUTCAR
  - iter_N/train.xyz (追加新数据)
```

#### 步骤 4: NEP 训练

```
输入:
  - iter_N/train.xyz (更新后的训练集)
  - nep.in (来自配置)

处理:
  1. 创建 nep_train 目录
  2. 写入 train.xyz 和 nep.in
  3. 提交 NEP 训练作业
  4. 等待完成

输出:
  - iter_N/nep_train/nep.txt
  - iter_N/nep.txt (复制)
```

#### 步骤 5: 更新活跃集

```
输入:
  - iter_N/train.xyz (更新后的训练集)
  - iter_N/nep.txt (新训练的模型)

处理:
  1. 计算描述符投影
  2. 执行 MaxVol 算法
  3. 生成新的活跃集

输出:
  - iter_N/active_set.asi
```

#### 步骤 6: 准备下一轮

```
处理:
  1. 创建 iter_{N+1}/ 目录
  2. 复制 train.xyz, nep.txt, active_set.asi
  3. 创建 GPUMD 任务目录
  4. 复制必要文件

输出:
  - iter_{N+1}/train.xyz
  - iter_{N+1}/nep.txt
  - iter_{N+1}/active_set.asi
  - iter_{N+1}/gpumd/<condition_id>/
```

## 文件命名规范

### 迭代目录

```
work/
├── iter_0/          # 初始化
├── iter_1/          # 第一轮迭代
├── iter_2/          # 第二轮迭代
└── ...
```

### 标准文件

每个迭代目录包含：

```
iter_N/
├── nep.txt              # NEP 模型
├── train.xyz            # 训练数据（累积）
├── active_set.asi       # 活跃集逆矩阵
├── active_set.xyz       # 活跃集结构（可选，分析用）
├── large_gamma.xyz      # GPUMD 收集的高 Gamma 结构
├── to_add.xyz           # 待 DFT 标注的结构
├── DONE                 # 完成标记
├── gpumd/               # GPUMD 探索
│   ├── 300K_NVT/
│   │   ├── model.xyz
│   │   ├── nep.txt
│   │   ├── active_set.asi
│   │   ├── run.in
│   │   ├── job.sh
│   │   ├── extrapolation_dump.xyz
│   │   └── DONE
│   └── ...
├── vasp/                # VASP DFT 计算
│   ├── task_0000/
│   │   ├── POSCAR
│   │   ├── INCAR
│   │   ├── POTCAR
│   │   ├── KPOINTS
│   │   ├── job.sh
│   │   ├── OUTCAR
│   │   └── DONE
│   └── ...
└── nep_train/           # NEP 训练
    ├── train.xyz
    ├── nep.in
    ├── job.sh
    ├── nep.txt
    └── DONE
```

## 任务状态管理

### DONE 文件机制

- 每个作业完成后应创建 `DONE` 文件
- 程序通过检测 `DONE` 文件判断作业是否完成
- 作业脚本应在最后添加：`touch DONE`

### 作业提交

- 使用 `subprocess.run()` 在作业目录执行提交命令
- 提交命令通过配置文件指定（如 `qsub job.sh`）
- 支持任意作业调度系统（PBS, SLURM, etc.）

### 超时处理

- 每种任务类型有独立的超时设置
- 超时后程序会记录警告并继续（可根据需要调整）

## 配置文件设计

### 路径解析规则

- **相对路径**: 基于 `work_dir` 解析
- **绝对路径**: 直接使用
- 示例:
  - `./input/INCAR` → `{work_dir}/input/INCAR`
  - `/data/INCAR` → `/data/INCAR`

### 内容 vs 路径

| 配置项 | 类型 | 说明 |
|--------|------|------|
| `vasp.incar_file` | 路径 | 指向文件 |
| `vasp.job_script` | 内容 | 多行字符串 |
| `nep.input_content` | 内容 | 多行字符串 |
| `gpumd.run_in_content` | 内容 | 多行字符串 |

### 验证规则

配置加载时会自动验证：
- 必需的输入文件是否存在
- GPUMD `run_in_content` 是否包含 `compute_extrapolation`
- 路径是否有效

## 扩展开发

### 添加新的任务类型

1. 在 `IterationManager` 中添加新方法
2. 在配置模块中添加配置类
3. 在 `run_iteration()` 中调用

### 添加新的选择策略

1. 在 `maxvol.py` 中实现选择函数
2. 在 `IterationManager.select_structures()` 中调用
3. 在配置中添加相关参数

### 自定义日志格式

修改 `initialize.py` 中的 `setup_logger()` 函数。

## 调试技巧

### 检查配置

```bash
python -m nep_auto.config config.yaml
```

### 仅运行初始化

```bash
python -m nep_auto.initialize config.yaml
```

### 从特定迭代继续

```bash
python -m nep_auto.main config.yaml --start-iter 3
```

### 查看日志

```bash
tail -f work/active_learning.log
```

### 手动测试 MaxVol

```python
from nep_auto import select_active_set, read_trajectory

train = read_trajectory("train.xyz")
result, selected = select_active_set(train, "nep.txt")
print(f"选中 {len(selected)} 个结构")
```

## 性能优化

### 批处理大小

- `batch_size` 控制 MaxVol 算法的批处理大小
- 默认 10000，对于大系统可以适当增加
- 内存占用约: `batch_size × descriptor_dim × 8 bytes`

### 并行作业

- GPUMD 多个条件可以并行运行
- VASP 多个结构可以并行计算
- 作业调度系统自动管理并行度

### Gamma 阈值调优

- `gamma_tol` 太小会导致选择过多结构
- `gamma_tol` 太大会导致选择过少结构
- 推荐值: 1.001 - 1.01

## 常见问题

### Q: 如何处理程序中断？

A: 使用 `--start-iter` 参数从中断的迭代继续：
```bash
python -m nep_auto.main config.yaml --start-iter 5
```

### Q: 如何跳过某个步骤？

A: 可以手动修改迭代管理器，或在配置中设置相应参数为空。

### Q: 如何自定义作业脚本？

A: 在配置文件中直接修改 `job_script` 字段（多行字符串）。

### Q: 如果 VASP 计算失败怎么办？

A: 程序会记录警告并跳过失败的结构。可以手动修复后重新运行。

### Q: 如何更改收敛criteria？

A: 修改 `select_structures()` 方法中的逻辑，当 `len(selected) == 0` 时收敛。

## 贡献指南

欢迎提交 Issue 和 Pull Request！

开发时请遵循：
- 使用类型注解
- 函数文档字符串使用中文
- 变量名使用英文
- 遵循 PEP 8 代码风格
