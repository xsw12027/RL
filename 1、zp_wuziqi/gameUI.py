import pygame
from config import *


class BoardUI:
    def __init__(self, game):
        self.game = game
        self.screen = None
        self.font = None
        self.small_font = None
        self.highlight_pos = None
        self.init_pygame()
    
    def init_fonts(self):
        try:
            # 尝试加载中文字体
            font_names = ["simhei.ttf", "msyh.ttc", "Arial Unicode MS", 
                          "DejaVuSans.ttf", "Arial.ttf"]
            
            for font_name in font_names:
                try:
                    self.font = pygame.font.Font(font_name, 36)
                    self.small_font = pygame.font.Font(font_name, 24)
                    if self.font and self.small_font:
                        return
                except:
                    continue
            
            # 使用系统字体
            self.font = pygame.font.SysFont(None, 36)
            self.small_font = pygame.font.SysFont(None, 24)
            
        except Exception as e:
            print(f"Font loading failed: {e}")
            self.font = pygame.font.Font(None, 36)
            self.small_font = pygame.font.Font(None, 24)
    
    def draw_board(self):
        if self.screen is None:
            return
        
        # 填充背景
        self.screen.fill(BOARD_COLOR)
        
        # 绘制棋盘网格线
        for i in range(self.game.board_size):
            # 横线
            pygame.draw.line(
                self.screen, LINE_COLOR,
                (MARGIN, MARGIN + i * GRID_SIZE),
                (MARGIN + (self.game.board_size - 1) * GRID_SIZE, MARGIN + i * GRID_SIZE),
                2
            )
            # 竖线
            pygame.draw.line(
                self.screen, LINE_COLOR,
                (MARGIN + i * GRID_SIZE, MARGIN),
                (MARGIN + i * GRID_SIZE, MARGIN + (self.game.board_size - 1) * GRID_SIZE),
                2
            )
        
        # 绘制棋盘上的定位点（天元和星）
        star_points = []
        if self.game.board_size >= 15:
            star_points = [(3, 3), (3, 11), (7, 7), (11, 3), (11, 11)]
        elif self.game.board_size >= 9:
            star_points = [(2, 2), (2, 6), (4, 4), (6, 2), (6, 6)]
        
        for r, c in star_points:
            x = MARGIN + c * GRID_SIZE
            y = MARGIN + r * GRID_SIZE
            pygame.draw.circle(self.screen, LINE_COLOR, (x, y), 5)
    
    def draw_pieces(self):
        if self.screen is None:
            return
        
        board = self.game.get_board()
        for row in range(self.game.board_size):
            for col in range(self.game.board_size):
                player = board[row][col]
                
                if player != 0:
                    x = MARGIN + col * GRID_SIZE
                    y = MARGIN + row * GRID_SIZE
                    
                    # 绘制棋子
                    color = BLACK if player == 1 else WHITE
                    pygame.draw.circle(self.screen, color, (x, y), PIECE_RADIUS)
                    
                    # 为白棋添加黑色边框
                    if player == 2:
                        pygame.draw.circle(self.screen, BLACK, (x, y), PIECE_RADIUS, 2)
        
        # 高亮显示最后一步
        if self.game.last_move:
            row, col = self.game.last_move
            x = MARGIN + col * GRID_SIZE
            y = MARGIN + row * GRID_SIZE
            pygame.draw.circle(self.screen, RED, (x, y), 8)
    
    def draw_ui(self):
        if self.screen is None:
            return
        
        # 当前玩家提示
        current_player = self.game.get_current_player()
        player_text = "黑棋回合" if current_player == 1 else "白棋回合"
        player_color = BLACK if current_player == 1 else WHITE
        
        # 创建半透明背景
        ui_bg = pygame.Surface((WINDOW_WIDTH, 100), pygame.SRCALPHA)
        ui_bg.fill((255, 255, 255, 180))
        self.screen.blit(ui_bg, (0, WINDOW_HEIGHT - 100))
        
        # 绘制玩家回合文本
        if self.font:
            text_surface = self.font.render(player_text, True, player_color)
            text_rect = text_surface.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT - 80))
            self.screen.blit(text_surface, text_rect)
        
        # 绘制游戏状态
        if self.game.is_game_over():
            winner = self.game.get_winner()
            if winner == 0:
                status_text = "平局"
                status_color = BLUE
            else:
                winner_text = "黑棋胜利" if winner == 1 else "白棋胜利"
                status_text = f"游戏结束 - {winner_text}"
                status_color = RED
            
            if self.font:
                status_surface = self.font.render(status_text, True, status_color)
                status_rect = status_surface.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT - 40))
                self.screen.blit(status_surface, status_rect)
        
        # 绘制操作提示
        hint_text = "鼠标点击落子 | R键重置 | T键切换 | ESC键退出"
        if self.small_font:
            hint_surface = self.small_font.render(hint_text, True, GRAY)
            hint_rect = hint_surface.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT - 20))
            self.screen.blit(hint_surface, hint_rect)
        
        # 绘制鼠标悬停预览
        if self.highlight_pos and not self.game.is_game_over():
            row, col = self.highlight_pos
            if self.game.board[row][col] == 0:  # 只有空位才显示预览
                x = MARGIN + col * GRID_SIZE
                y = MARGIN + row * GRID_SIZE
                
                # 半透明预览棋子
                preview_color = (0, 0, 0, 100) if current_player == 1 else (255, 255, 255, 100)
                preview_surface = pygame.Surface((PIECE_RADIUS * 2, PIECE_RADIUS * 2), pygame.SRCALPHA)
                pygame.draw.circle(preview_surface, preview_color, (PIECE_RADIUS, PIECE_RADIUS), PIECE_RADIUS)
                self.screen.blit(preview_surface, (x - PIECE_RADIUS, y - PIECE_RADIUS))
    
    def get_board_position(self, mouse_pos):
        x, y = mouse_pos
        
        col = round((x - MARGIN) / GRID_SIZE)
        row = round((y - MARGIN) / GRID_SIZE)
        
        row_legal = (0 <= row < self.game.board_size)
        col_legal = (0 <= col < self.game.board_size)
        grid_x_legal = (abs(x - (MARGIN + col * GRID_SIZE)) <= GRID_SIZE / 3)
        grid_y_legal = (abs(y - (MARGIN + row * GRID_SIZE)) <= GRID_SIZE / 3)
        
        if row_legal and col_legal and grid_x_legal and grid_y_legal:
            return (row, col)
        
        return None
    
    def update_highlight(self, mouse_pos):
        self.highlight_pos = self.get_board_position(mouse_pos)
    
    def handle_click(self, mouse_pos):
        pos = self.get_board_position(mouse_pos)
        if pos:
            row, col = pos
            return self.game.make_move(row, col)
        return False
    
    def draw(self):
        if self.screen is None:
            return
        
        self.draw_board()
        self.draw_pieces()
        self.draw_ui()
    
    def reset(self):
        self.game.reset()
    
    def init_pygame(self):
        pygame.init()
        pygame.display.set_caption(f"{BOARD_SIZE}×{BOARD_SIZE} 五子棋 - AlphaZero")
        
        # 创建窗口
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        
        # 初始化字体
        self.init_fonts()