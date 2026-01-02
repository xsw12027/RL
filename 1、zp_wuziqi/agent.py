import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# 导入配置
import config
from environment import *


# 改进的残差块
class ImprovedResidualBlock(nn.Module):
    """改进的残差块：使用3×3卷积，更高效"""
    def __init__(self, channels):
        super().__init__()
        #3x3卷积保持特征图尺寸
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)  # inplace减少内存使用
        
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out


# 改进的主训练网络
class AplhaZeroNet(nn.Module):
    """增强版AlphaZero网络，集成了价值近似和策略近似"""
    def __init__(self, board_size=config.BOARD_SIZE, channels=config.CHANNELS, num_residual_blocks=config.RES_BLOCK_NUM):
        super().__init__()
        
        # 输入层，使用3×3卷积，padding=1保持尺寸
        self.conv_input = nn.Conv2d(3, channels, kernel_size=3, padding=1, bias=False)
        self.bn_input = nn.BatchNorm2d(channels)
        
        # 残差塔
        self.residual_tower = nn.Sequential(
            *[ImprovedResidualBlock(channels) for _ in range(num_residual_blocks)]
        )
        
        # 策略头
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(32 * board_size * board_size, board_size * board_size)
        )
        
        # 价值头
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(32 * board_size * board_size, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
            nn.Tanh()
        )
        
        # 权重初始化
        self._init_weights()
    
    def _init_weights(self):
        """使用Kaiming初始化改进训练稳定性"""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # 输入处理
        x = x.detach().clone().to(dtype=torch.float32)
        x = F.relu(self.bn_input(self.conv_input(x)), inplace=True)
        
        # 残差塔
        x = self.residual_tower(x)
        
        # 双头输出
        policy = F.softmax(self.policy_head(x), dim=1)
        value = self.value_head(x)
        
        # 裁剪价值输出避免极端值
        value = torch.clamp(value, -config.VALUE_CLAMP, config.VALUE_CLAMP)
        
        return policy, value


# 蒙特卡洛树搜索MCTS
class MCTS:
    def __init__(self, model, simulations):
        self.model = model
        self.simulations = simulations
        self.Q = {}  # 动作价值
        self.N = {}  # 访问次数
        self.P = {}  # 先验策略
        self.V = {}  # 状态价值缓存
        
        # 缓存机制，减少重复计算
        self.state_cache = {}
    
    def clear(self):
        """清空搜索树"""
        self.Q.clear()
        self.N.clear()
        self.P.clear()
        self.V.clear()
        self.state_cache.clear()
    
    def get_state_key(self, board):
        """获取状态键值，用于缓存"""
        return tuple(board.board.flatten())
    
    def expand_node(self, board):
        """扩展新节点，修正价值方向"""
        s = self.get_state_key(board)
        
        # 检查游戏是否结束
        winner = board.check_win()
        if winner != 0:
            if winner == 3:  # 平局
                return 0.0
            elif winner == board.current_player:  # 当前玩家赢
                return 1.0
            else:  # 当前玩家输
                return -1.0
        
        # 获取网络预测
        state_tensor = torch.from_numpy(board.get_state()).unsqueeze(0).to(config.DEVICE_SELFPLAY, non_blocking=True)
        
        with torch.no_grad():
            policy, value = self.model(state_tensor)
        
        policy = policy.cpu().numpy()[0]
        value = value.item()
        
        # 创建合法动作掩码
        legal_moves = board.get_legal_moves()
        mask = np.zeros(config.BOARD_SIZE * config.BOARD_SIZE, dtype=np.float32)
        mask[legal_moves] = 1.0
        
        # 应用掩码并归一化
        policy = policy * mask
        if policy.sum() > 0:
            policy /= policy.sum()
        else:
            # 如果没有合法动作的概率，均匀分配
            policy = mask / mask.sum()
        
        # 存储到搜索树
        self.P[s] = policy
        self.V[s] = value  # 缓存状态价值
        
        return value
    
    def select_action(self, s, board):
        """选择UCT分数最高的动作"""
        best_u, best_a = -float('inf'), -1
        ns = math.sqrt(self.N.get(s, 0) + 1e-8)
        legal_moves = board.get_legal_moves()
        
        for a in legal_moves:
            # PUCT算法
            q_value = self.Q.get((s, a), 0.0)
            p_value = self.P[s][a] if s in self.P else 1.0 / len(legal_moves)
            n_value = self.N.get((s, a), 0)
            
            # 探索项计算
            u = q_value + config.C_PUCT * p_value * ns / (1 + n_value)
            
            if u > best_u:
                best_u = u
                best_a = a
        
        return best_a
    
    def search(self, board, depth=0):
        """执行一次MCTS搜索，修正价值传递"""
        # 深度限制，防止无限递归
        if depth >= config.MAX_DEPTH:
            # 深度限制时，使用网络评估
            if hasattr(board, 'get_state'):
                state_tensor = torch.from_numpy(board.get_state()).unsqueeze(0).to(config.DEVICE_SELFPLAY)
                with torch.no_grad():
                    _, value = self.model(state_tensor)
                return value.item()
            return 0.0
        
        s = self.get_state_key(board)
        
        # 如果节点未扩展，进行扩展并返回价值估计
        if s not in self.P:
            leaf_value = self.expand_node(board)
            # 缓存叶子节点价值
            if s not in self.V:
                self.V[s] = leaf_value
            return leaf_value
        
        # 选择动作
        action = self.select_action(s, board)
        if action == -1:  # 没有合法动作
            return 0.0
        
        # 执行动作
        next_board = board.copy()
        x, y = divmod(action, config.BOARD_SIZE)
        next_board.move(x, y)
        
        # 递归搜索，得到子节点的价值
        child_value = self.search(next_board, depth + 1)
        
        # 关键修正：从当前玩家视角，子节点价值需要取反
        # 因为子节点是对手行动后的状态，价值是相反的
        value = -child_value
        
        # 更新统计信息
        sa = (s, action)
        self.N[sa] = self.N.get(sa, 0) + 1
        self.Q[sa] = value  # 存储从当前视角的价值
        
        # 更新状态访问次数
        self.N[s] = self.N.get(s, 0) + 1
        
        return value
    
    def get_policy(self, board, temperature=1.0):
        """获取搜索后的策略分布"""
        s = self.get_state_key(board)
        legal_moves = board.get_legal_moves()
        
        # 获取访问次数
        counts = np.zeros(config.BOARD_SIZE * config.BOARD_SIZE, dtype=np.float32)
        total_visits = 0
        
        for a in legal_moves:
            visits = self.N.get((s, a), 0)
            counts[a] = visits
            total_visits += visits
        
        # 避免除零错误
        if total_visits == 0:
            # 均匀分布
            counts[legal_moves] = 1.0 / len(legal_moves)
            return counts
        
        # 应用温度参数
        if temperature <= 0.1:
            # 低温时直接选择最优
            max_count = counts.max()
            counts = (counts == max_count).astype(np.float32)
        else:
            counts = counts ** (1.0 / temperature)
        
        # 归一化
        counts_sum = counts.sum()
        if counts_sum > 0:
            counts /= counts_sum
        else:
            counts[legal_moves] = 1.0 / len(legal_moves)
        
        return counts
    
    def search_root(self, current_board_numpy, current_player):
        """旧接口兼容"""
        root_board = FastBoard()
        root_board.board = current_board_numpy.copy()
        root_board.current_player = current_player
        
        for _ in range(self.simulations):
            self.search(root_board.copy())
        
        s_key = tuple(root_board.board.flatten())
        counts = np.zeros(config.BOARD_SIZE * config.BOARD_SIZE)
        legal = root_board.get_legal_moves()
        for a in legal:
            counts[a] = self.N.get((s_key, a), 0)
        return counts


# 优化的工作线程
_worker_net = None

def init_worker(shared_state_dict):
    """初始化工作线程"""
    global _worker_net
    _worker_net = AplhaZeroNet().to(config.DEVICE_SELFPLAY)
    _worker_net.load_state_dict(shared_state_dict)
    _worker_net.eval()

def self_play_worker(simulations, temperature_threshold=config.TEMPERATURE_THRESHOLD, dirichlet_alpha=config.DIRICHLET_ALPHA):
    """执行自对弈，添加步数限制"""
    global _worker_net
    
    # 随机化模拟次数
    actual_sim = max(100, simulations + np.random.randint(-20, 20))
    
    board = Board()
    mcts = MCTS(_worker_net, actual_sim)
    
    states, policies, players = [], [], []
    
    step = 0
    max_steps = config.BOARD_SIZE * config.BOARD_SIZE
    
    while True:
        # 防止无限循环
        if step >= max_steps:
            # 棋盘已满，平局
            return list(zip(states, policies, [0.0] * len(states)))
        
        # 清空MCTS树，为新的决策做准备
        mcts.clear()
        
        # 执行MCTS搜索
        for _ in range(mcts.simulations):
            mcts.search(board.copy())
        
        # 生成策略分布
        pi = mcts.get_policy(
            board, 
            temperature=1.0 if step < temperature_threshold else 0.1
        )
        
        # 添加Dirichlet噪声（前几步）
        if step < 8:
            legal_moves = board.get_legal_moves()
            noise = np.random.dirichlet([dirichlet_alpha] * len(legal_moves))
            
            # 混合噪声
            pi_legal = pi[legal_moves]
            pi_legal = (1 - config.DIRICHLET_WEIGHT) * pi_legal + config.DIRICHLET_WEIGHT * noise
            
            # 重新归一化
            if pi_legal.sum() > 0:
                pi_legal /= pi_legal.sum()
            else:
                pi_legal = np.ones_like(pi_legal) / len(pi_legal)
            
            pi[legal_moves] = pi_legal
        
        # 存储数据
        states.append(board.get_state())
        policies.append(pi)
        players.append(board.current_player)
        
        # 选择动作
        legal_moves = board.get_legal_moves()
        
        if step < temperature_threshold:
            # 温度较高时使用随机采样
            action_probs = pi[legal_moves]
            if action_probs.sum() > 0:
                action_probs /= action_probs.sum()
            else:
                action_probs = np.ones_like(action_probs) / len(action_probs)
            
            action_idx = np.random.choice(len(legal_moves), p=action_probs)
            action = legal_moves[action_idx]
        else:
            # 温度较低时选择最优动作
            # 只在合法动作中选择
            legal_pi = pi[legal_moves]
            action_idx = np.argmax(legal_pi)
            action = legal_moves[action_idx]
        
        # 执行动作
        x, y = divmod(action, config.BOARD_SIZE)
        board.move(x, y)
        
        # 检查游戏是否结束
        winner = board.check_win()
        if winner != 0:
            # 计算每个状态的价值
            values = []
            for player in players:
                if winner == 3:  # 平局
                    values.append(0.0)
                else:
                    # 赢家得到+1，输家得到-1
                    values.append(1.0 if winner == player else -1.0)
            
            return list(zip(states, policies, values))
        
        step += 1