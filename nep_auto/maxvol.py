"""
MaxVol 算法模块

本模块提供基于 MaxVol (最大体积) 算法的主动学习结构选择功能。
核心功能包括：
- 描述符投影计算 (B_projection)
- MaxVol 算法选择活跃集
- Gamma 值计算（外推等级评估）
- Active Set Inverse (ASI) 文件读写

参考文献:
    Goreinov S., Oseledets, I., Savostyanov, D., Tyrtyshnikov, E., Zamarashkin, N.
    "How to find a good submatrix". Matrix Methods: Theory, Algorithms And
    Applications (2010): 247-256.

代码来源:
    部分基于 https://github.com/AndreiChertkov/teneva
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import lu, solve_triangular
from typing import Literal
from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm

# 尝试导入 ASE 和 PyNEP
try:
    from ase import Atoms
    from ase.io import read as ase_read, write as ase_write
except ImportError:
    Atoms = None

try:
    from pynep.calculate import NEP
    from pynep.io import load_nep, dump_nep
except ImportError:
    NEP = None


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class ActiveSetResult:
    """MaxVol 算法的输出结果"""

    inverse_dict: dict[str, NDArray[np.float64]]
    """按元素类型分类的活跃集逆矩阵 {元素符号: 逆矩阵}"""

    structure_indices: list[int]
    """被选中结构在原始轨迹中的索引"""

    active_set_dict: dict[str, NDArray[np.float64]]
    """按元素类型分类的活跃集矩阵 {元素符号: 描述符矩阵}"""


@dataclass
class DescriptorProjectionResult:
    """描述符投影计算的输出结果"""

    projection_dict: dict[str, NDArray[np.float64]]
    """按元素类型分类的 B 投影矩阵 {元素符号: (N_atoms, D) 矩阵}"""

    structure_index_dict: dict[str, NDArray[np.int64]]
    """按元素类型分类的结构索引 {元素符号: 原子所属结构索引数组}"""


# =============================================================================
# Core MaxVol Algorithm (CPU Version)
# =============================================================================


def _maxvol_core(
    A: NDArray[np.float64],
    gamma_tol: float = 1.001,
    max_iter: int = 1000,
) -> NDArray[np.int64]:
    """
    MaxVol 核心算法：从高矩阵中选择最大体积子矩阵。

    算法通过迭代交换行，使得选中的子矩阵具有最大的行列式（体积）。
    这保证了选中的行能够最大程度地张成原始矩阵的列空间。

    参数:
        A: 输入的高矩阵，形状为 (n, r)，要求 n > r
        gamma_tol: 收敛精度参数，应 >= 1.0
            - 等于 1.0 时会迭代直到完全收敛
            - 大于 1.0 时算法更快但精度略低（推荐 1.01 - 1.1）
        max_iter: 允许的最大迭代次数

    返回:
        被选中行的索引数组，长度为 r

    异常:
        ValueError: 当输入矩阵不是高矩阵时抛出
    """
    n, r = A.shape

    if n <= r:
        raise ValueError(f"输入矩阵必须是高矩阵 (n > r)，当前: n={n}, r={r}")

    # LU decomposition for initialization
    P, L, U = lu(A, check_finite=False)
    selected_indices = P[:, :r].argmax(axis=0)

    # Compute coefficient matrix B = A @ A[I]^(-1)
    Q = solve_triangular(U, A.T, trans=1, check_finite=False)
    B = solve_triangular(
        L[:r, :], Q, trans=1, check_finite=False, unit_diagonal=True, lower=True
    ).T

    # Iterative optimization
    for _ in range(max_iter):
        # Find the element with maximum absolute value
        max_pos = np.abs(B).argmax()
        i, j = divmod(max_pos, r)
        current_gamma = np.abs(B[i, j])

        # Check convergence
        if current_gamma <= gamma_tol:
            break

        # Swap row
        selected_indices[j] = i

        # Update coefficient matrix (Sherman-Morrison formula)
        bj = B[:, j]
        bi = B[i, :].copy()
        bi[j] -= 1.0
        B -= np.outer(bj, bi / B[i, j])

    return selected_indices


def _compute_pinv(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    计算矩阵的伪逆。

    使用较大的条件数阈值 (1e-8) 以避免 GPUMD 使用 float 类型时的数值问题。

    参数:
        matrix: 输入矩阵

    返回:
        伪逆矩阵
    """
    return np.linalg.pinv(matrix, rcond=1e-8)


# =============================================================================
# Batch MaxVol Algorithm
# =============================================================================


def compute_maxvol(
    A: NDArray[np.float64],
    struct_index: NDArray[np.int64],
    gamma_tol: float = 1.001,
    max_iter: int = 1000,
    batch_size: int | None = None,
    n_refinement: int = 10,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """
    执行 MaxVol 算法，支持批量处理和迭代细化。

    对于大规模数据，使用批量处理策略：
    1. 将数据分成多个批次
    2. 每个批次与之前的结果合并后执行 MaxVol
    3. 最后进行多轮细化确保收敛

    参数:
        A: 描述符矩阵，形状为 (N, D)
        struct_index: 每个环境对应的结构索引
        gamma_tol: MaxVol 收敛阈值
        max_iter: 单次 MaxVol 的最大迭代次数
        batch_size: 批处理大小,None 表示一次性处理
        n_refinement: 批处理后的细化迭代次数

    返回:
        (选中的描述符矩阵, 选中的结构索引)
    """
    # Single batch mode
    if batch_size is None:
        selected = _maxvol_core(A, gamma_tol, max_iter)
        return A[selected], struct_index[selected]

    # Multi-batch mode
    n_batches = int(np.ceil(len(A) / batch_size))
    batch_splits = np.array_split(np.arange(len(A)), n_batches)

    # Stage 1: Cumulative MaxVol
    A_selected: NDArray[np.float64] | None = None
    index_selected: NDArray[np.int64] | None = None

    for i, batch_indices in enumerate(batch_splits):
        if A_selected is None:
            # First batch
            A_joint = A[batch_indices]
            index_joint = struct_index[batch_indices]
        else:
            # Subsequent batches: merge with selected
            A_joint = np.vstack([A_selected, A[batch_indices]])
            index_joint = np.hstack([index_selected, struct_index[batch_indices]])

        selected = _maxvol_core(A_joint, gamma_tol, max_iter)
        prev_len = 0 if A_selected is None else len(A_selected)
        A_selected = A_joint[selected]
        index_selected = index_joint[selected]
        n_added = (selected >= prev_len).sum()
        print(f"Batch {i + 1}/{n_batches}: added {n_added} environments")

    # Stage 2: Refinement
    assert A_selected is not None and index_selected is not None

    for ii in range(n_refinement):
        inv_matrix = _compute_pinv(A_selected)
        gamma_matrix = np.abs(A_selected @ inv_matrix)
        large_gamma_mask = gamma_matrix > gamma_tol
        max_gamma = np.max(gamma_matrix)

        print(
            f"Refinement {ii + 1}: {large_gamma_mask.sum()} envs exceed threshold, max gamma = {max_gamma:.4f}"
        )

        if max_gamma < gamma_tol:
            print("Refinement done")
            break

        # Add environments exceeding threshold to candidates
        A_joint = np.vstack([A_selected, A[large_gamma_mask.any(axis=1)]])
        index_joint = np.hstack(
            [index_selected, struct_index[large_gamma_mask.any(axis=1)]]
        )
        selected = _maxvol_core(A_joint, gamma_tol, max_iter)
        A_selected = A_joint[selected]
        index_selected = index_joint[selected]

    return A_selected, index_selected


# =============================================================================
# Descriptor Computation
# =============================================================================


def compute_descriptor_projection(
    trajectory: list[Atoms],
    nep_file: str | Path,
    show_progress: bool = True,
) -> DescriptorProjectionResult:
    """
    计算轨迹中所有原子的 NEP 描述符投影 (B_projection)。

    描述符投影是 NEP 势函数中每个原子局部环境的低维表示，
    用于评估模型的外推程度。

    参数:
        trajectory: ASE Atoms 对象列表
        nep_file: NEP 势函数文件路径 (nep.txt)
        show_progress: 是否显示进度条

    返回:
        描述符投影结果，包含按元素分类的投影矩阵和结构索引

    异常:
        ImportError: 当 PyNEP 未安装时抛出
    """
    if NEP is None:
        raise ImportError("请先安装 PyNEP: pip install pynep")

    nep_file = Path(nep_file)
    calc = NEP(str(nep_file))

    # Parse element list from NEP file
    with open(nep_file) as f:
        first_line = f.readline()
        parts = first_line.split()
        n_types = int(parts[1])
        elements = parts[2 : 2 + n_types]  # Format: nep4 N_types elem1 elem2 ... elemN
    print(f"Elements in NEP potential: {elements}")

    # Initialize storage
    projection_dict: dict[str, list[NDArray]] = {elem: [] for elem in elements}
    struct_index_dict: dict[str, list[int]] = {elem: [] for elem in elements}

    # Iterate over trajectory
    iterator = (
        tqdm(
            enumerate(trajectory), total=len(trajectory), desc="Computing B_projection"
        )
        if show_progress
        else enumerate(trajectory)
    )

    for struct_idx, atoms in iterator:
        calc.calculate(atoms, ["B_projection"])
        B_proj = calc.results["B_projection"]

        for atom_proj, symbol in zip(B_proj, atoms.get_chemical_symbols()):
            projection_dict[symbol].append(atom_proj)
            struct_index_dict[symbol].append(struct_idx)

    # Convert to NumPy arrays
    projection_dict_arr = {}
    struct_index_dict_arr = {}
    print("Descriptor matrix shapes:")
    for elem in elements:
        if len(projection_dict[elem]) > 0:
            projection_dict_arr[elem] = np.vstack(projection_dict[elem])
            struct_index_dict_arr[elem] = np.array(
                struct_index_dict[elem], dtype=np.int64
            )
            print(f"  {elem}: {projection_dict_arr[elem].shape}")

            # Verify the matrix is tall
            n, d = projection_dict_arr[elem].shape
            if n < d:
                raise ValueError(
                    f"元素 {elem} 的环境数量不足: 需要至少 {d} 个，当前只有 {n} 个"
                )

    return DescriptorProjectionResult(
        projection_dict=projection_dict_arr,
        structure_index_dict=struct_index_dict_arr,
    )


def compute_gamma(
    trajectory: list[Atoms],
    nep_file: str | Path,
    asi_file: str | Path,
    show_progress: bool = True,
) -> list[Atoms]:
    """
    计算轨迹中每个原子的 Gamma 值（外推等级）。

    Gamma 值表示原子局部环境相对于训练集的外推程度：
    - Gamma ≈ 1: 环境在训练集覆盖范围内
    - Gamma > 1: 环境超出训练集，存在外推风险
    - Gamma >> 1: 严重外推，模型预测不可靠

    计算结果存储在每个 Atoms 对象的 arrays["gamma"] 中。

    参数:
        trajectory: ASE Atoms 对象列表
        nep_file: NEP 势函数文件路径
        asi_file: Active Set Inverse 文件路径
        show_progress: 是否显示进度条

    返回:
        更新后的轨迹（原地修改，同时返回引用）
    """
    if NEP is None:
        raise ImportError("请先安装 PyNEP: pip install pynep")

    calc = NEP(str(nep_file))
    active_set_inv = read_asi_file(asi_file)

    iterator = tqdm(trajectory, desc="Computing gamma") if show_progress else trajectory

    for atoms in iterator:
        # Initialize gamma array
        atoms.arrays["gamma"] = np.zeros(len(atoms))

        # Compute descriptor projection
        calc.calculate(atoms, ["B_projection"])
        B_proj = calc.results["B_projection"]

        # Compute gamma by element
        symbols = atoms.get_chemical_symbols()
        for elem, inv_matrix in active_set_inv.items():
            atom_indices = [i for i, sym in enumerate(symbols) if sym == elem]
            if len(atom_indices) == 0:
                continue

            # gamma = |B @ A^(-1)|_max (max per atom)
            g = B_proj[atom_indices] @ inv_matrix
            g = np.max(np.abs(g), axis=1)
            atoms.arrays["gamma"][atom_indices] = g

    return trajectory


# =============================================================================
# Active Set Generation
# =============================================================================


def generate_active_set(
    descriptor_result: DescriptorProjectionResult,
    gamma_tol: float = 1.001,
    batch_size: int = 10000,
    write_asi: bool = True,
    asi_output_path: str | Path = "active_set.asi",
) -> ActiveSetResult:
    """
    使用 MaxVol 算法从描述符投影中生成活跃集。

    活跃集是训练数据中最具代表性的原子环境子集，
    其张成的空间能够覆盖所有训练数据的描述符。

    参数:
        descriptor_result: 描述符投影计算结果
        gamma_tol: MaxVol 收敛阈值
        batch_size: 批处理大小
        write_asi: 是否将结果写入 ASI 文件
        asi_output_path: ASI 文件输出路径

    返回:
        活跃集结果
    """
    print("Running MaxVol algorithm...")
    active_set_dict: dict[str, NDArray] = {}
    all_struct_indices: list[int] = []

    for elem, B_proj in descriptor_result.projection_dict.items():
        print(f"\nProcessing element: {elem}")
        A_selected, index_selected = compute_maxvol(
            B_proj,
            descriptor_result.structure_index_dict[elem],
            gamma_tol=gamma_tol,
            batch_size=batch_size,
        )
        active_set_dict[elem] = A_selected
        all_struct_indices.extend(index_selected.tolist())
        print(f"Active set shape: {A_selected.shape}")

    # Deduplicate and sort structure indices
    structure_indices = sorted(set(all_struct_indices))

    # Compute inverse matrices
    print("\nComputing active set inverse...")
    inverse_dict = {elem: _compute_pinv(A) for elem, A in active_set_dict.items()}

    # Save ASI file
    if write_asi:
        print(f"Saving active set inverse to: {asi_output_path}")
        write_asi_file(inverse_dict, asi_output_path)

    return ActiveSetResult(
        inverse_dict=inverse_dict,
        structure_indices=structure_indices,
        active_set_dict=active_set_dict,
    )


# =============================================================================
# ASI File I/O
# =============================================================================


def write_asi_file(
    active_set_inv: dict[str, NDArray[np.float64]],
    file_path: str | Path = "active_set.asi",
) -> None:
    """
    将活跃集逆矩阵保存到 ASI 文件。

    ASI (Active Set Inverse) 文件格式:
    ```
    元素符号 行数 列数
    矩阵元素1
    矩阵元素2
    ...
    ```

    参数:
        active_set_inv: 按元素分类的逆矩阵字典
        file_path: 输出文件路径
    """
    with open(file_path, "w") as f:
        for elem, matrix in active_set_inv.items():
            f.write(f"{elem} {matrix.shape[0]} {matrix.shape[1]}\n")
            for val in matrix.flatten():
                f.write(f"{val}\n")


def read_asi_file(file_path: str | Path) -> dict[str, NDArray[np.float64]]:
    """
    从 ASI 文件读取活跃集逆矩阵。

    参数:
        file_path: ASI 文件路径

    返回:
        按元素分类的逆矩阵字典
    """
    result: dict[str, NDArray] = {}

    with open(file_path, "r") as f:
        while True:
            line1 = f.readline()
            if not line1:
                break

            parts = line1.split()
            elem, rows, cols = parts[0], int(parts[1]), int(parts[2])

            data = [float(f.readline()) for _ in range(rows * cols)]
            result[elem] = np.array(data).reshape((rows, cols))

    return result


# =============================================================================
# High-Level Selection Functions
# =============================================================================


def select_active_set(
    trajectory: list[Atoms],
    nep_file: str | Path,
    asi_output_path: str | Path = "active_set.asi",
    gamma_tol: float = 1.001,
    batch_size: int = 10000,
) -> tuple[ActiveSetResult, list[Atoms]]:
    """
    从训练轨迹中选择活跃集。

    这是 select_active.py 的函数化版本。

    参数:
        trajectory: 训练集轨迹
        nep_file: NEP 势函数文件路径
        asi_output_path: ASI 文件输出路径
        gamma_tol: MaxVol 收敛阈值
        batch_size: 批处理大小

    返回:
        (活跃集结果, 被选中的结构列表)
    """
    # Compute descriptor projection
    descriptor_result = compute_descriptor_projection(trajectory, nep_file)

    # Generate active set
    active_set = generate_active_set(
        descriptor_result,
        gamma_tol=gamma_tol,
        batch_size=batch_size,
        write_asi=True,
        asi_output_path=asi_output_path,
    )

    # Extract selected structures
    selected_structures = [trajectory[i] for i in active_set.structure_indices]

    return active_set, selected_structures


def select_extension_structures(
    train_trajectory: list[Atoms],
    candidate_trajectory: list[Atoms],
    nep_file: str | Path,
    gamma_tol: float = 1.001,
    batch_size: int = 10000,
) -> list[Atoms]:
    """
    从候选结构中选择需要标注的新结构。

    这是 select_extend.py 的函数化版本。
    算法合并训练集和候选集，执行 MaxVol，然后只返回来自候选集的结构。

    参数:
        train_trajectory: 当前训练集
        candidate_trajectory: 高 Gamma 候选结构
        nep_file: NEP 势函数文件路径
        gamma_tol: MaxVol 收敛阈值
        batch_size: 批处理大小

    返回:
        被选中的新结构列表（仅来自候选集）
    """
    train_size = len(train_trajectory)
    merged_trajectory = train_trajectory + candidate_trajectory

    # Compute descriptor projection
    descriptor_result = compute_descriptor_projection(merged_trajectory, nep_file)

    # Generate active set (without writing ASI file)
    active_set = generate_active_set(
        descriptor_result,
        gamma_tol=gamma_tol,
        batch_size=batch_size,
        write_asi=False,
    )

    # Keep only structures from candidate set
    new_structures = [
        merged_trajectory[i] for i in active_set.structure_indices if i >= train_size
    ]

    print(
        f"\nSelected {len(new_structures)} new structures from {len(candidate_trajectory)} candidates"
    )
    return new_structures


def filter_high_gamma_structures(
    trajectory: list[Atoms],
    nep_file: str | Path,
    asi_file: str | Path,
    gamma_min: float = 1.0,
    gamma_max: float = float("inf"),
) -> list[Atoms]:
    """
    根据 Gamma 值筛选结构。

    这是 select_gamma.py 的函数化版本。

    参数:
        trajectory: 待筛选的轨迹
        nep_file: NEP 势函数文件路径
        asi_file: Active Set Inverse 文件路径
        gamma_min: Gamma 下限阈值
        gamma_max: Gamma 上限阈值

    返回:
        满足 gamma_min < max_gamma < gamma_max 的结构列表
    """
    # Compute gamma values
    compute_gamma(trajectory, nep_file, asi_file)

    # Filter
    filtered = []
    for atoms in trajectory:
        max_gamma = atoms.arrays["gamma"].max()
        if gamma_min < max_gamma < gamma_max:
            filtered.append(atoms)

    print(
        f"Filtered {len(filtered)} high-gamma structures from {len(trajectory)} total"
    )
    return filtered


# =============================================================================
# Convenience I/O Functions
# =============================================================================


def read_trajectory(
    file_path: str | Path,
    format: Literal["nep", "xyz", "auto"] = "auto",
) -> list[Atoms]:
    """
    读取轨迹文件。

    参数:
        file_path: 轨迹文件路径
        format: 文件格式 ("nep", "xyz", "auto")

    返回:
        Atoms 对象列表
    """
    file_path = Path(file_path)

    if format == "auto":
        format = "nep" if file_path.suffix == ".xyz" else "xyz"

    if format == "nep" and load_nep is not None:
        try:
            return load_nep(str(file_path))
        except Exception:
            pass

    return ase_read(str(file_path), index=":")


def write_trajectory(
    trajectory: list[Atoms],
    file_path: str | Path,
    format: Literal["nep", "xyz", "auto"] = "auto",
) -> None:
    """
    写入轨迹文件。

    参数:
        trajectory: Atoms 对象列表
        file_path: 输出文件路径
        format: 文件格式 ("nep", "xyz", "auto")
    """
    file_path = Path(file_path)

    if format == "auto":
        format = "nep"

    if format == "nep" and dump_nep is not None:
        try:
            dump_nep(str(file_path), trajectory)
            return
        except Exception:
            pass

    ase_write(str(file_path), trajectory)
