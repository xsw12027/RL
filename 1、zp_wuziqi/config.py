import torch

# 此处为整个项目所需要的配置信息
RESUME = True
CHECKPOINT_FILE = "./models/checkpoint.pth"
BUFFER_FILE = "./models/replay_buffer.pkl"

BOARD_SIZE = 15                              # 棋盘大小
WIN_COUNT = 5                                # 获胜连子数目
TRUNATED_RATE = 0.85                         # 在棋盘有多少比例的位置已经落子时，结束对局


NUM_WORKERS = 16                            # 工作进程数
SELF_PLAY_GAMES = 32                        # 游戏数量
BASE_SIMULATION = 300                       # 中等模拟次数保证质量
MAX_SIMULATION = 500                        # 最大模拟次数
MCTS_SIMULATIONS = 800                      # 测试时的思考强度

# 训练批次
BATCH_SIZE = 512                            # 批次大小
EPOCHS_PER_UPDATE = 2                       # 训练轮次

# 缓冲区配置
REPLAY_BUFFER_SIZE = 200000                 # 缓冲区容量
MIN_BUFFER_TO_TRAIN = 1000                  # 最小样本数

# 网络架构优化
LEARNING_RATE = 3e-4                      # 学习率
WEIGHT_DECAY = 1e-4                         # 权重衰减
RES_BLOCK_NUM = 6                           # 残差块数量
KERNEL_SIZE = 3                             # 卷积核大小
CHANNELS = 192                              # 通道数
PADDING = (KERNEL_SIZE - 1) // 2            # 卷积填充

IS_TEST = True                             # 是否为测试模式

# MCTS参数
DIRICHLET_ALPHA = 0.15
DIRICHLET_WEIGHT = 0.25
TEMPERATURE_THRESHOLD = 25                  # 温度阈值
C_PUCT = 2.5                                # 增加探索率，避免过早收敛
MAX_DEPTH = 120                             # MCTS最大深度限制

MAX_ITERS = 600                             # 最大训练迭代次数
MODEL_DIR = "./models"

# 训练参数
VALUE_LOSS_WEIGHT = 1.0                     # 价值损失的权重
GRAD_CLIP = 1.0                             # 梯度裁剪阈值
LABEL_SMOOTHING = 0.1                       # 标签平滑参数
VALUE_CLAMP = 0.95                          # 价值输出裁剪，避免极端值

# UI的颜色配置
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BOARD_COLOR = (222, 184, 135)
LINE_COLOR = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 120, 255)
GRAY = (200, 200, 200)

# 学习率调度 - 更合理的调度策略
LR_MILESTONES = [560,575,590]              # 学习率衰减里程碑（每1/3, 2/3, 5/6训练过程）
LR_SCHEDULER_GAMMA = 0.5                   # 学习率衰减因子
USE_AMP = True                             # 启用自动混合精度训练

# UI参数
GRID_SIZE = 40
MARGIN = 50
PIECE_RADIUS = 18
FPS = 60
WINDOW_WIDTH = 2 * MARGIN + (BOARD_SIZE - 1) * GRID_SIZE
WINDOW_HEIGHT = 2 * MARGIN + (BOARD_SIZE - 1) * GRID_SIZE + 120


# 设备配置
DEVICE_TRAIN = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE_SELFPLAY = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE_TEST = torch.device("cuda" if torch.cuda.is_available() else "cpu")