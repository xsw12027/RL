import pickle
import numpy as np
from collections import deque

# 导入配置信息
import config


# 棋盘逻辑
class Board:
    def __init__(self):
        self.board = np.zeros((config.BOARD_SIZE, config.BOARD_SIZE), dtype=np.int8)
        self.board_size = config.BOARD_SIZE
        self.current_player = 1
        self.last_move = (-1, 1)
        self.move_history = []
        self.zobrist_hash = 0
        self._init_zobrist()
    
    def _init_zobrist(self):
        """初始化Zobrist哈希表"""
        np.random.seed(42)
        self.zobrist_table = np.random.randint(
            0, 2**63, size=(config.BOARD_SIZE, config.BOARD_SIZE, 3), dtype=np.uint64
        )
        self.hash = 0
    
    def _update_hash(self, x, y, player):
        """更新Zobrist哈希"""
        self.hash ^= self.zobrist_table[x, y, player]
    
    def copy(self):
        new_board = Board()
        new_board.board = self.board.copy()
        new_board.current_player = self.current_player
        new_board.last_move = self.last_move
        new_board.move_history = self.move_history.copy()
        new_board.hash = self.hash
        return new_board
    
    def get_state(self):
        """获取当前状态表示"""
        me = (self.board == self.current_player).astype(np.float32)
        opp = (self.board == 3 - self.current_player).astype(np.float32)
        color = np.full((config.BOARD_SIZE, config.BOARD_SIZE), self.current_player / 2.0, dtype=np.float32)
        return np.stack([me, opp, color])
    
    def move(self, x, y):
        """执行落子"""
        if not (0 <= x < config.BOARD_SIZE and 0 <= y < config.BOARD_SIZE):
            return False
        
        if self.board[x][y] != 0:
            return False
        
        self.board[x][y] = self.current_player
        self._update_hash(x, y, self.current_player)
        self.last_move = (x, y)
        self.move_history.append((x, y, self.current_player))
        self.current_player = 3 - self.current_player
        return True
    
    def undo(self):
        """撤销上一步"""
        if not self.move_history:
            return False
        
        x, y, player = self.move_history.pop()
        self.board[x][y] = 0
        self._update_hash(x, y, player)  # 两次异或恢复原状
        self.current_player = player
        self.last_move = self.move_history[-1][:2] if self.move_history else (-1, -1)
        return True
    
    def check_truncated(self):
        """检查是否应该提前结束"""
        if config.IS_TEST:
            return np.all(self.board != 0)
        else:
            proportion = np.count_nonzero(self.board == 0) / self.board.size
            return (1 - proportion) >= config.TRUNATED_RATE
    
    def check_win(self):
        """检查是否有玩家获胜"""
        lx, ly = self.last_move
        player = self.board[lx, ly]
        
        if lx == -1 or player == 0:
            return 0
        
        # 检查四个方向
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        
        for dx, dy in directions:
            count = 1
            
            # 正方向
            for k in range(1, config.WIN_COUNT):
                nx, ny = lx + dx * k, ly + dy * k
                if 0 <= nx < config.BOARD_SIZE and 0 <= ny < config.BOARD_SIZE and self.board[nx, ny] == player:
                    count += 1
                else:
                    break
            
            # 负方向
            for k in range(1, config.WIN_COUNT):
                nx, ny = lx - dx * k, ly - dy * k
                if 0 <= nx < config.BOARD_SIZE and 0 <= ny < config.BOARD_SIZE and self.board[nx, ny] == player:
                    count += 1
                else:
                    break
            
            if count >= config.WIN_COUNT:
                return player
        
        if self.check_truncated():
            return 3
        
        return 0
    
    def get_legal_moves(self):
        """获取合法走法"""
        # 优先考虑棋盘中心区域
        indices = np.argwhere(self.board == 0)
        
        if len(indices) == 0:
            return [0]
        
        # 如果有上一步，优先考虑周围位置
        if self.last_move[0] != -1:
            lx, ly = self.last_move
            moves = []
            
            # 收集周围2格内的位置
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    nx, ny = lx + dx, ly + dy
                    if 0 <= nx < config.BOARD_SIZE and 0 <= ny < config.BOARD_SIZE and self.board[nx, ny] == 0:
                        moves.append(nx * config.BOARD_SIZE + ny)
            
            # 如果有周围位置，优先返回
            if moves:
                return moves
        
        # 否则返回所有空位
        moves = indices[:, 0] * config.BOARD_SIZE + indices[:, 1]
        return moves.tolist()
    
    def get_heuristic_moves(self, distance=2):
        """获取启发式走法（只考虑周围有棋子的位置）"""
        if len(self.move_history) < 5:
            return self.get_legal_moves()
        
        # 收集所有已有棋子的位置
        occupied = np.argwhere(self.board != 0)
        if len(occupied) == 0:
            return self.get_legal_moves()
        
        moves = set()
        for ox, oy in occupied:
            for dx in range(-distance, distance + 1):
                for dy in range(-distance, distance + 1):
                    nx, ny = ox + dx, oy + dy
                    if (0 <= nx < config.BOARD_SIZE and 0 <= ny < config.BOARD_SIZE and 
                        self.board[nx, ny] == 0):
                        moves.add(nx * config.BOARD_SIZE + ny)
        
        return list(moves) if moves else self.get_legal_moves()
    
    def display(self):
        """打印棋盘（用于调试）"""
        symbols = {0: '.', 1: 'X', 2: 'O'}
        print("  " + " ".join(str(i).rjust(2) for i in range(config.BOARD_SIZE)))
        for i in range(config.BOARD_SIZE):
            row = [symbols[self.board[i][j]] for j in range(config.BOARD_SIZE)]
            print(f"{i:2} " + " ".join(row))


# 用于MCTS搜索的快速棋盘类
class FastBoard:
    """用于MCTS搜索的快速棋盘类"""
    def __init__(self):
        self.board = np.zeros((config.BOARD_SIZE, config.BOARD_SIZE), dtype=np.int8)
        self.current_player = 1
        self.last_move = (-1, -1)
    
    def copy(self):
        new_board = FastBoard()
        new_board.board = self.board.copy()
        new_board.current_player = self.current_player
        new_board.last_move = self.last_move
        return new_board
    
    def move(self, x, y):
        if not (0 <= x < config.BOARD_SIZE and 0 <= y < config.BOARD_SIZE):
            return False
        
        if self.board[x][y] != 0:
            return False
        
        self.board[x][y] = self.current_player
        self.last_move = (x, y)
        self.current_player = 3 - self.current_player
        return True
    
    def get_state(self):
        """获取当前状态表示 - 与Board类保持一致"""
        me = (self.board == self.current_player).astype(np.float32)
        opp = (self.board == 3 - self.current_player).astype(np.float32)
        color = np.full((config.BOARD_SIZE, config.BOARD_SIZE), self.current_player / 2.0, dtype=np.float32)
        return np.stack([me, opp, color])
    
    def get_legal_moves(self):
        indices = np.argwhere(self.board == 0)
        if len(indices) == 0:
            return [0]
        moves = indices[:, 0] * config.BOARD_SIZE + indices[:, 1]
        return moves.tolist()
    
    def check_win(self):
        """简化版的获胜检查，仅用于MCTS"""
        lx, ly = self.last_move
        if lx == -1:
            return 0
        
        player = self.board[lx, ly]
        if player == 0:
            return 0
        
        # 检查四个方向
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        
        for dx, dy in directions:
            count = 1
            
            # 正方向
            for k in range(1, config.WIN_COUNT):
                nx, ny = lx + dx * k, ly + dy * k
                if 0 <= nx < config.BOARD_SIZE and 0 <= ny < config.BOARD_SIZE and self.board[nx, ny] == player:
                    count += 1
                else:
                    break
            
            # 负方向
            for k in range(1, config.WIN_COUNT):
                nx, ny = lx - dx * k, ly - dy * k
                if 0 <= nx < config.BOARD_SIZE and 0 <= ny < config.BOARD_SIZE and self.board[nx, ny] == player:
                    count += 1
                else:
                    break
            
            if count >= config.WIN_COUNT:
                return player
        
        # 检查是否平局
        if np.all(self.board != 0):
            return 3
        
        return 0