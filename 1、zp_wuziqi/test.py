import pygame
import sys
import os
import time
import numpy as np
import torch
import config
from agent import *
from environment import *
from gameUI import *
from game import *


def load_latest_model(model_dir):
    """加载最新的模型"""
    models = [f for f in os.listdir(model_dir) if f.endswith(".pth")]
    if not models:
        return None
    
    # 按修改时间排序
    models.sort(key=lambda x: os.path.getmtime(os.path.join(model_dir, x)))
    
    # 优先选择"strong"或"best"模型
    for prefix in ["strong", "best", "final"]:
        for model in reversed(models):
            if model.startswith(prefix):
                return model
    
    # 否则返回最新的
    return models[-1]


def main_test():
    print(f"Hardware: {config.DEVICE_TEST}")
    print(f"Board Size: {config.BOARD_SIZE}x{config.BOARD_SIZE}")
    
    # 确保模型目录存在
    if not os.path.exists(config.MODEL_DIR):
        print(f"Error: {config.MODEL_DIR} does not exist.")
        return
    
    # 加载模型
    model_name = load_latest_model(config.MODEL_DIR)
    if not model_name:
        print("Error: No model found in the directory.")
        return
    
    model_path = os.path.join(config.MODEL_DIR, model_name)
    print(f"Loading Model: {model_name}")
    
    model_net = AplhaZeroNet().to(config.DEVICE_TEST)
    try:
        model_net.load_state_dict(torch.load(model_path, map_location=config.DEVICE_TEST, weights_only=False))
        model_net.eval()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Model Load Failed: {e}")
        return
    
    # 初始化MCTS
    mcts = MCTS(model_net, config.MCTS_SIMULATIONS)
    
    # 初始化游戏和UI
    game = Game(config.BOARD_SIZE)
    board_ui = BoardUI(game)
    clock = pygame.time.Clock()
    
    # 游戏状态
    running = True
    human_turn = False
    game.reset()
    
    pygame.display.set_caption(f"AlphaZero 五子棋 - {model_name}")
    
    # 添加AI思考深度显示
    thinking_depth = 0
    
    while running:
        # 事件处理
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    game.reset()
                    human_turn = True
                    mcts.clear()
                    print("Game Reset")
                elif event.key == pygame.K_t:
                    # 切换玩家
                    human_turn = not human_turn
                    print(f"Turn switched. Now {'Human' if human_turn else 'AI'} plays.")
                elif event.key == pygame.K_s:
                    # 显示棋盘状态
                    game.display()
            
            elif event.type == pygame.MOUSEMOTION:
                board_ui.update_highlight(event.pos)
            
            elif event.type == pygame.MOUSEBUTTONDOWN and not game.is_game_over():
                if event.button == 1 and human_turn:
                    pos = board_ui.get_board_position(event.pos)
                    if pos:
                        r, c = pos
                        if game.make_move(r, c):
                            print(f"Human Move: ({r}, {c})")
                            human_turn = False
        
        # AI思考逻辑
        if not human_turn and not game.is_game_over():
            board_ui.draw()
            pygame.display.flip()
            
            print("AI Thinking... ", end="", flush=True)
            think_start = time.time()
            
            # 使用MCTS搜索
            mcts.clear()
            
            # 创建FastBoard对象，并设置其状态
            current_fast_board = FastBoard()
            current_fast_board.board = game.board.copy()
            current_fast_board.current_player = game.current_player
            current_fast_board.last_move = game.last_move if game.last_move else (-1, -1)
            
            for i in range(config.MCTS_SIMULATIONS):
                mcts.search(current_fast_board.copy())
                thinking_depth = i + 1
                
                # 每100次模拟更新一次显示
                if i % 100 == 0:
                    board_ui.draw()
                    pygame.display.flip()
            
            # 获取策略 - 使用FastBoard对象
            pi = mcts.get_policy(current_fast_board, temperature=0.1)
            action = np.argmax(pi)
            
            r, c = divmod(action, config.BOARD_SIZE)
            think_time = time.time() - think_start
            
            if game.make_move(r, c):
                print(f"AI Move: ({r}, {c}) in {think_time:.1f}s, Depth: {thinking_depth}")
                human_turn = True
            else:
                print("AI tried to move an invalid position.")
                # 选择次优位置 - 修正：获取Game类的合法走法
                # 创建一个临时棋盘对象来获取合法走法
                temp_board = Board()
                temp_board.board = game.board.copy()
                temp_board.current_player = game.current_player
                legal_moves = temp_board.get_legal_moves()
                
                if legal_moves and len(legal_moves) > 0:
                    # 从pi中筛选合法走法的概率
                    legal_probs = pi[legal_moves]
                    if legal_probs.sum() > 0:
                        legal_probs = legal_probs / legal_probs.sum()
                        action_idx = np.random.choice(len(legal_moves), p=legal_probs)
                        action = legal_moves[action_idx]
                    else:
                        action = np.random.choice(legal_moves)
                    
                    r, c = divmod(action, config.BOARD_SIZE)
                    if game.make_move(r, c):
                        print(f"AI Fallback Move: ({r}, {c})")
                        human_turn = True
                    else:
                        print("AI fallback also failed, game might be stuck.")
                        # 如果还是失败，切换回合
                        human_turn = True
        
        # 绘制界面
        board_ui.draw()
        
        # 显示额外信息
        if not human_turn and not game.is_game_over():
            # 显示"思考中..."
            font = pygame.font.Font(None, 30)
            text = font.render(f"AI思考中... ({thinking_depth}/{config.MCTS_SIMULATIONS})", True, config.RED)
            board_ui.screen.blit(text, (10, config.WINDOW_HEIGHT - 40))
        
        pygame.display.flip()
        clock.tick(config.FPS)
    
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main_test()