import numpy as np
from config import *


class Game:
    def __init__(self, board_size=BOARD_SIZE):
        self.board_size = board_size
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        self.current_player = 1
        self.game_over = False
        self.winner = 0
        self.moves = []
        self.last_move = None
        self.move_count = 0
    
    def reset(self):
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        self.current_player = 1
        self.game_over = False
        self.winner = 0
        self.moves = []
        self.last_move = None
        self.move_count = 0
    
    def check_win(self, row, col):
        player = self.board[row, col]
        directions = [
            [(0, 1), (0, -1)],    # 水平
            [(1, 0), (-1, 0)],    # 垂直
            [(1, 1), (-1, -1)],   # 主对角线
            [(1, -1), (-1, 1)]    # 副对角线
        ]
        
        for dir_pair in directions:
            count = 1
            
            for dx, dy in dir_pair:
                dr, dc = row + dx, col + dy
                
                while (0 <= dr < self.board_size and 
                       0 <= dc < self.board_size and 
                       self.board[dr, dc] == player):
                    count += 1
                    dr += dx
                    dc += dy
                
                if count >= WIN_COUNT:
                    return True
        
        return False
    
    def make_move(self, row, col):
        if self.game_over or self.board[row, col] != 0:
            return False
        
        self.board[row, col] = self.current_player
        self.moves.append((row, col, self.current_player))
        self.last_move = (row, col)
        self.move_count += 1
        
        if self.check_win(row, col):
            self.game_over = True
            self.winner = self.current_player
            print(f"Player {self.current_player} wins!")
        elif self.move_count == self.board_size * self.board_size:
            self.game_over = True
            self.winner = 0
            print("Game ended in a draw.")
        else:
            self.current_player = 3 - self.current_player
        
        return True
    
    def get_board(self):
        return self.board.copy()
    
    def get_current_player(self):
        return self.current_player
    
    def is_game_over(self):
        return self.game_over
    
    def get_winner(self):
        return self.winner
    
    def display(self):
        """显示棋盘（用于调试）"""
        symbols = {0: '.', 1: 'X', 2: 'O'}
        print("\n  " + " ".join(str(i).rjust(2) for i in range(self.board_size)))
        for i in range(self.board_size):
            row = [symbols[self.board[i][j]] for j in range(self.board_size)]
            print(f"{i:2} " + " ".join(row))
        print()