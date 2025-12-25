"""
主程序入口

完整的主动学习流程：
1. 初始化工作空间（迭代 0）
2. 运行迭代循环（迭代 1, 2, 3, ...）
3. 直到收敛或达到最大迭代次数
"""

import sys
from pathlib import Path

from .config import load_config, print_config_summary
from .initialize import setup_logger, initialize_workspace
from .iteration import IterationManager


def main(config_file: str, start_iter: int = 0) -> None:
    """
    主函数：运行完整的主动学习流程

    参数:
        config_file: 配置文件路径
        start_iter: 起始迭代编号（0=从初始化开始，>0=从指定迭代继续）
    """
    # 加载配置
    print("=" * 80)
    print("NEP 主动学习框架")
    print("=" * 80)
    print(f"\n加载配置文件: {config_file}")

    try:
        config = load_config(config_file)
    except Exception as e:
        print(f"配置加载失败: {e}")
        sys.exit(1)

    # 打印配置摘要
    print_config_summary(config)

    # 设置日志
    logger = setup_logger(config.global_config.log_file)

    # 步骤 0: 初始化（如果从头开始）
    if start_iter == 0:
        logger.info("=" * 80)
        logger.info("开始主动学习流程")
        logger.info("=" * 80)

        try:
            initialize_workspace(config, logger)
        except Exception as e:
            logger.error(f"初始化失败: {e}")
            sys.exit(1)

        # 初始化完成，开始迭代
        start_iter = 1

    # 创建迭代管理器
    iteration_manager = IterationManager(config, logger)

    # 运行迭代循环
    max_iterations = config.global_config.max_iterations
    current_iter = start_iter

    while current_iter <= max_iterations:
        try:
            # 运行一次迭代
            should_continue = iteration_manager.run_iteration(current_iter)

            if not should_continue:
                # 收敛或失败
                logger.info("\n" + "=" * 80)
                logger.info(f"主动学习在迭代 {current_iter} 结束")
                logger.info("=" * 80)
                break

            # 继续下一轮
            current_iter += 1

        except KeyboardInterrupt:
            logger.warning("\n用户中断程序")
            logger.info(f"当前进度: 迭代 {current_iter}")
            logger.info(
                f"重启命令: python -m nep_auto.main {config_file} --start-iter {current_iter}"
            )
            sys.exit(0)

        except Exception as e:
            logger.error(f"迭代 {current_iter} 发生异常: {e}")
            logger.exception("详细错误信息:")
            sys.exit(1)

    else:
        # 达到最大迭代次数
        logger.info("\n" + "=" * 80)
        logger.info(f"达到最大迭代次数 ({max_iterations})")
        logger.info("=" * 80)

    logger.info("\n主动学习流程完成！")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="NEP 主动学习框架",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 从头开始运行
  python -m nep_auto.main config.yaml

  # 从指定迭代继续运行（比如中断后恢复）
  python -m nep_auto.main config.yaml --start-iter 5

  # 仅初始化（不运行迭代）
  python -m nep_auto.initialize config.yaml
        """,
    )

    parser.add_argument(
        "config",
        type=str,
        help="配置文件路径（YAML 格式）",
    )

    parser.add_argument(
        "--start-iter",
        type=int,
        default=0,
        help="起始迭代编号（0=从初始化开始，>0=从指定迭代继续，默认: 0）",
    )

    args = parser.parse_args()

    # 检查配置文件是否存在
    if not Path(args.config).exists():
        print(f"错误: 配置文件不存在: {args.config}")
        sys.exit(1)

    # 运行主程序
    main(args.config, args.start_iter)
