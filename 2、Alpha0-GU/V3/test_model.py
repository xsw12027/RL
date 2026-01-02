"""
test_model.py - Standalone Gomoku AI Tester (AlphaZero V3)

Integrates config, game logic, UI, network, and MCTS into a single file.
Allows the user to select any model from the 'models_v3' folder.
"""

import os
import sys
import pygame
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

# ==================== Configuration ====================
BOARD_SIZE = 15
WIN_COUNT = 5
GRID_SIZE = 40
MARGIN = 50
PIECE_RADIUS = 18

WINDOW_WIDTH = 2 * MARGIN + (BOARD_SIZE - 1) * GRID_SIZE
WINDOW_HEIGHT = 2 * MARGIN + (BOARD_SIZE - 1) * GRID_SIZE + 120

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (0, 120, 255)
GRAY = (200, 200, 200)
BOARD_COLOR = (222, 184, 135)   # Classic wooden board color
LINE_COLOR = (0, 0, 0)

MODEL_DIR = "models_v3"
MCTS_SIMS = 800                 # AI thinking strength
INPUT_CHANNELS = 4              # [Me, Opponent, Color, LastMove]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==================== Neural Network (SE-ResNet) ====================
class SEResidualBlock(nn.Module):
    """Squeeze-and-Excitation residual block."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        b, c, _, _ = out.size()
        y = self.avg_pool(out).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        out = out * y
        out += residual
        return F.relu(out)


class AlphaZeroNet(nn.Module):
    """AlphaZero-style network with policy and value heads."""
    def __init__(self, num_blocks=10, channels=128):
        super().__init__()
        self.conv_in = nn.Conv2d(INPUT_CHANNELS, channels, 3, padding=1, bias=False)
        self.bn_in = nn.BatchNorm2d(channels)
        self.blocks = nn.Sequential(*[SEResidualBlock(channels) for _ in range(num_blocks)])
        self.conv_p = nn.Conv2d(channels, 2, 1)
        self.bn_p = nn.BatchNorm2d(2)
        self.fc_p = nn.Linear(2 * BOARD_SIZE * BOARD_SIZE, BOARD_SIZE * BOARD_SIZE)
        self.conv_v = nn.Conv2d(channels, 1, 1)
        self.bn_v = nn.BatchNorm2d(1)
        self.fc_v1 = nn.Linear(BOARD_SIZE * BOARD_SIZE, 128)
        self.fc_v2 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.bn_in(self.conv_in(x)))
        x = self.blocks(x)

        # Policy head
        p = F.relu(self.bn_p(self.conv_p(x))).view(x.size(0), -1)
        p = self.fc_p(p)
        p = F.softmax(p, dim=1)

        # Value head
        v = F.relu(self.bn_v(self.conv_v(x))).view(x.size(0), -1)
        v = F.relu(self.fc_v1(v))
        v = torch.tanh(self.fc_v2(v))

        return p, v


# ==================== Game Logic ====================
class GomokuGame:
    """Core game logic for 15×15 Gomoku."""
    def __init__(self, board_size=BOARD_SIZE):
        self.board_size = board_size
        self.reset()

    def reset(self):
        """Reset the board and game state."""
        self.board = np.zeros((self.board_size, self.board_size), dtype=int)
        self.current_player = 1          # 1: Black, 2: White
        self.game_over = False
        self.winner = 0
        self.last_move = None

    def make_move(self, row, col):
        """Place a stone at (row, col) for the current player."""
        if self.game_over or self.board[row, col] != 0:
            return False

        self.board[row, col] = self.current_player
        self.last_move = (row, col)

        if self._check_win(row, col):
            self.game_over = True
            self.winner = self.current_player
        elif np.count_nonzero(self.board) == self.board_size * self.board_size:
            self.game_over = True
            self.winner = 0  # Draw
        else:
            self.current_player = 3 - self.current_player

        return True

    def _check_win(self, row, col):
        """Check if the last move created five-in-a-row."""
        player = self.board[row, col]
        directions = [
            [(0, 1), (0, -1)],   # horizontal
            [(1, 0), (-1, 0)],   # vertical
            [(1, 1), (-1, -1)],  # main diagonal
            [(1, -1), (-1, 1)]   # anti-diagonal
        ]

        for dir_pair in directions:
            count = 1
            for dx, dy in dir_pair:
                r, c = row + dx, col + dy
                while (0 <= r < self.board_size and 0 <= c < self.board_size and
                       self.board[r, c] == player):
                    count += 1
                    r += dx
                    c += dy
            if count >= 5:
                return True
        return False

    def get_board(self):
        return self.board.copy()

    def get_current_player(self):
        return self.current_player

    def is_game_over(self):
        return self.game_over

    def get_winner(self):
        return self.winner


# ==================== Fast Board for MCTS ====================
class FastBoard:
    """Lightweight board used only by MCTS (no UI dependencies)."""
    def __init__(self):
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        self.current_player = 1
        self.last_move = (-1, -1)

    def copy(self):
        new = FastBoard()
        new.board = self.board.copy()
        new.current_player = self.current_player
        new.last_move = self.last_move
        return new

    def get_state(self):
        """Return network input planes."""
        me = (self.board == self.current_player).astype(np.float32)
        opp = (self.board == 3 - self.current_player).astype(np.float32)
        color = np.full((BOARD_SIZE, BOARD_SIZE), self.current_player / 2.0, dtype=np.float32)

        last_move_map = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        lx, ly = self.last_move
        if lx >= 0 and ly >= 0:
            last_move_map[lx, ly] = 1.0

        return np.stack([me, opp, color, last_move_map])

    def move(self, row, col):
        if self.board[row, col] != 0:
            return False
        self.board[row, col] = self.current_player
        self.last_move = (row, col)
        self.current_player = 3 - self.current_player
        return True

    def check_win(self):
        lx, ly = self.last_move
        if lx < 0:
            return 0
        player = self.board[lx, ly]
        if player == 0:
            return 0

        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for dx, dy in directions:
            count = 1
            for k in range(1, WIN_COUNT):
                nx, ny = lx + k * dx, ly + k * dy
                if not (0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE and self.board[nx, ny] == player):
                    break
                count += 1
            for k in range(1, WIN_COUNT):
                nx, ny = lx - k * dx, ly - k * dy
                if not (0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE and self.board[nx, ny] == player):
                    break
                count += 1
            if count >= WIN_COUNT:
                return player
        if np.all(self.board != 0):
            return 3  # Draw
        return 0

    def get_legal_moves(self):
        indices = np.argwhere(self.board == 0)
        return indices[:, 0] * BOARD_SIZE + indices[:, 1]


# ==================== MCTS ====================
class MCTS:
    """Simple pure MCTS (no Dirichlet noise needed for evaluation)."""
    def __init__(self, net, simulations):
        self.net = net
        self.simulations = simulations
        self.Q = {}
        self.N = {}
        self.P = {}
        self.C_PUCT = 2.0

    def get_action_counts(self, board_numpy, current_player, last_move):
        """Run MCTS from the current position and return visit counts."""
        root = FastBoard()
        root.board = board_numpy.copy()
        root.current_player = current_player
        root.last_move = last_move if last_move is not None else (-1, -1)

        for _ in range(self.simulations):
            self._search(root.copy())

        s = tuple(root.board.flatten())
        counts = np.zeros(BOARD_SIZE * BOARD_SIZE, dtype=np.float32)
        legal = root.get_legal_moves()
        for a in legal:
            counts[a] = self.N.get((s, a), 0)
        return counts

    def _search(self, board):
        s = tuple(board.board.flatten())
        winner = board.check_win()
        if winner != 0:
            return 0.0 if winner == 3 else -1.0

        if s not in self.P:
            state = torch.from_numpy(board.get_state()).unsqueeze(0).to(device)
            with torch.no_grad():
                p, v = self.net(state)
            p = p.cpu().numpy()[0]
            v = v.item()

            legal = board.get_legal_moves()
            mask = np.zeros(BOARD_SIZE * BOARD_SIZE)
            mask[legal] = 1
            p = p * mask
            p_sum = p.sum()
            if p_sum > 0:
                p /= p_sum
            else:
                p[legal] = 1.0 / len(legal)

            self.P[s] = p
            self.Q[s] = self.N[s] = 0
            return -v

        best_u = -float('inf')
        best_a = -1
        ns = math.sqrt(self.N[s] + 1)
        legal = board.get_legal_moves()

        for a in legal:
            u = self.Q.get((s, a), 0) + self.C_PUCT * self.P[s][a] * ns / (1 + self.N.get((s, a), 0))
            if u > best_u:
                best_u = u
                best_a = a

        board.move(best_a // BOARD_SIZE, best_a % BOARD_SIZE)
        v = self._search(board)

        sa = (s, best_a)
        self.N[sa] = self.N.get(sa, 0) + 1
        self.Q[sa] = (self.Q.get(sa, 0) * (self.N[sa] - 1) + v) / self.N[sa]
        self.N[s] += 1
        return -v


# ==================== UI with Pygame ====================
class GomokuUI:
    """Handles all drawing and mouse interaction."""
    def __init__(self, game):
        self.game = game
        self.screen = None
        self.font = None
        self.small_font = None
        self.highlight_pos = None
        self._init_pygame()

    def _init_pygame(self):
        pygame.init()
        pygame.display.set_caption("Gomoku AlphaZero V3 - 15×15")
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self._init_fonts()

    def _init_fonts(self):
        try:
            font_names = ['simhei', 'msyh', 'Arial Unicode MS', 'DejaVu Sans']
            for name in font_names:
                try:
                    self.font = pygame.font.SysFont(name, 36)
                    self.small_font = pygame.font.SysFont(name, 24)
                    if self.font and self.small_font:
                        return
                except:
                    continue
            self.font = pygame.font.Font(None, 36)
            self.small_font = pygame.font.Font(None, 24)
        except:
            self.font = pygame.font.Font(None, 36)
            self.small_font = pygame.font.Font(None, 24)

    def draw_board(self):
        self.screen.fill(BOARD_COLOR)
        for i in range(self.game.board_size):
            pygame.draw.line(self.screen, LINE_COLOR,
                             (MARGIN, MARGIN + i * GRID_SIZE),
                             (MARGIN + (BOARD_SIZE - 1) * GRID_SIZE, MARGIN + i * GRID_SIZE), 2)
            pygame.draw.line(self.screen, LINE_COLOR,
                             (MARGIN + i * GRID_SIZE, MARGIN),
                             (MARGIN + i * GRID_SIZE, MARGIN + (BOARD_SIZE - 1) * GRID_SIZE), 2)

        # Star points (for 15×15)
        star_points = [(3, 3), (3, 11), (11, 3), (11, 11), (7, 7)]
        for r, c in star_points:
            pygame.draw.circle(self.screen, LINE_COLOR,
                               (MARGIN + c * GRID_SIZE, MARGIN + r * GRID_SIZE), 6)

    def draw_pieces(self):
        board = self.game.get_board()
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                player = board[r, c]
                if player != 0:
                    color = BLACK if player == 1 else WHITE
                    center = (MARGIN + c * GRID_SIZE, MARGIN + r * GRID_SIZE)
                    pygame.draw.circle(self.screen, color, center, PIECE_RADIUS)
                    if player == 2:  # white border for white pieces
                        pygame.draw.circle(self.screen, BLACK, center, PIECE_RADIUS, 2)

        # Highlight last move
        if self.game.last_move:
            r, c = self.game.last_move
            pygame.draw.circle(self.screen, RED,
                               (MARGIN + c * GRID_SIZE, MARGIN + r * GRID_SIZE), 8)

    def draw_ui(self):
        current = self.game.get_current_player()
        player_text = "Black's turn" if current == 1 else "White's turn"
        player_color = BLACK if current == 1 else WHITE

        # Semi-transparent bottom bar
        bar = pygame.Surface((WINDOW_WIDTH, 100), pygame.SRCALPHA)
        bar.fill((255, 255, 255, 180))
        self.screen.blit(bar, (0, WINDOW_HEIGHT - 100))

        if self.font:
            txt = self.font.render(player_text, True, player_color)
            self.screen.blit(txt, txt.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT - 80)))

        if self.game.is_game_over():
            winner = self.game.get_winner()
            if winner == 0:
                status = "Game Over: Draw!"
                color = BLUE
            else:
                status = f"Game Over: {'Black' if winner == 1 else 'White'} Wins!"
                color = RED
            txt = self.font.render(status, True, color)
            self.screen.blit(txt, txt.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT - 40)))

        hint = "Click to place | R: Reset | ESC: Quit"
        if self.small_font:
            txt = self.small_font.render(hint, True, GRAY)
            self.screen.blit(txt, txt.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT - 20)))

        # Hover preview
        if self.highlight_pos:
            r, c = self.highlight_pos
            center = (MARGIN + c * GRID_SIZE, MARGIN + r * GRID_SIZE)
            preview_color = (0, 0, 0, 80) if current == 1 else (255, 255, 255, 80)
            s = pygame.Surface((PIECE_RADIUS * 2, PIECE_RADIUS * 2), pygame.SRCALPHA)
            pygame.draw.circle(s, preview_color, (PIECE_RADIUS, PIECE_RADIUS), PIECE_RADIUS)
            self.screen.blit(s, (center[0] - PIECE_RADIUS, center[1] - PIECE_RADIUS))

    def get_board_pos(self, mouse_pos):
        x, y = mouse_pos
        col = round((x - MARGIN) / GRID_SIZE)
        row = round((y - MARGIN) / GRID_SIZE)
        if (0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE and
                abs(x - (MARGIN + col * GRID_SIZE)) <= GRID_SIZE // 3 and
                abs(y - (MARGIN + row * GRID_SIZE)) <= GRID_SIZE // 3):
            return row, col
        return None

    def update_highlight(self, mouse_pos):
        self.highlight_pos = self.get_board_pos(mouse_pos)

    def handle_click(self, mouse_pos):
        pos = self.get_board_pos(mouse_pos)
        if pos:
            return self.game.make_move(*pos)
        return False

    def draw(self):
        self.draw_board()
        self.draw_pieces()
        self.draw_ui()
        pygame.display.flip()


# ==================== Main ====================
def select_model():
    """List all models in models_v3 and let the user choose one."""
    candidates = []
    if os.path.exists("checkpoint_v3.pth"):
        candidates.append(("Checkpoint", "checkpoint_v3.pth"))
    if os.path.isdir(MODEL_DIR):
        for f in os.listdir(MODEL_DIR):
            if f.endswith(".pth"):
                candidates.append((f, os.path.join(MODEL_DIR, f)))

    if not candidates:
        print("❌ No model files found.")
        sys.exit(1)

    print("\nAvailable models:")
    for i, (name, _) in enumerate(candidates, 1):
        print(f" [{i}] {name}")
    while True:
        try:
            choice = int(input("\nSelect model number: ")) - 1
            if 0 <= choice < len(candidates):
                return candidates[choice][1]
        except:
            pass
        print("Invalid choice, try again.")


def main():
    print(f"🚀 Gomoku AlphaZero V3 Tester | Device: {device}")

    # Choose who goes first
    print("\nWho moves first?")
    print(" [1] Human (Black)")
    print(" [2] AI (Black)")
    choice = input("Choose (1/2, default 1): ").strip()
    human_first = choice != "2"

    # Load model
    model_path = select_model()
    print(f"Loading model: {model_path}")

    net = AlphaZeroNet().to(device)
    try:
        state = torch.load(model_path, map_location=device, weights_only=False)
        if isinstance(state, dict) and 'net' in state:
            net.load_state_dict(state['net'])
        else:
            net.load_state_dict(state)
        net.eval()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model: {e}")
        sys.exit(1)

    mcts = MCTS(net, MCTS_SIMS)

    # Initialize game and UI
    game = GomokuGame()
    ui = GomokuUI(game)
    clock = pygame.time.Clock()

    game.reset()
    human_turn = human_first

    pygame.display.set_caption(f"Gomoku V3 - {'Human' if human_first else 'AI'} First")

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    game.reset()
                    human_turn = human_first
                    print("Game reset.")
                elif event.key == pygame.K_ESCAPE:
                    running = False
            elif event.type == pygame.MOUSEMOTION:
                ui.update_highlight(event.pos)
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if human_turn and not game.is_game_over():
                    if ui.handle_click(event.pos):
                        human_turn = False

        # AI move
        if not human_turn and not game.is_game_over():
            ui.draw()
            pygame.display.flip()

            # First move shortcut (center)
            if not human_first and game.last_move is None:
                center = BOARD_SIZE // 2
                game.make_move(center, center)
                print(f"AI places first stone at center ({center}, {center})")
                human_turn = True
                continue

            print(f"AI thinking ({MCTS_SIMS} simulations)... ", end="", flush=True)
            counts = mcts.get_action_counts(game.board, game.current_player, game.last_move)
            action = int(np.argmax(counts))
            row, col = divmod(action, BOARD_SIZE)
            if game.make_move(row, col):
                print(f"AI move: ({row}, {col})")
                human_turn = True
            else:
                print("AI illegal move (bug)")

        ui.draw()
        clock.tick(60)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()