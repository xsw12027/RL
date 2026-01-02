import os
import sys

# 将项目根目录添加到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入配置信息
import config

# 根据配置决定导入哪个模块
if config.IS_TEST:
    from test import main_test
else:
    from train import main_train


def main():
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    print(f"Mode: {'TEST' if config.IS_TEST else 'TRAIN'}")
    print(f"Board Size: {config.BOARD_SIZE}x{config.BOARD_SIZE}")
    print(f"Device: {config.DEVICE_TRAIN}")
    
    if config.IS_TEST:
        main_test()
    else:
        main_train()


if __name__ == "__main__":
    main()