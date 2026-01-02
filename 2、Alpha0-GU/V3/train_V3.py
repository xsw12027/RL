import os
import math
import time
import random
import pickle
from collections import deque

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.amp import autocast, GradScaler

# ==================== Configuration ====================
# Architecture Settings
NUM_BLOCKS = 10
CHANNELS = 128
INPUT_CHANNELS = 4  # [Player, Opponent, Color, LastMove]

# Policy Optimization
BATCH_SIZE = 4096
DIRICHLET_ALPHA = 0.1
DIRICHLET_WEIGHT = 0.25

REPLAY_BUFFER_SIZE = 30000
MIN_BUFFER_TO_TRAIN = 2000

# Search Quality
BASE_SIMULATIONS = 400
MAX_SIMULATIONS = 800
SELF_PLAY_GAMES = 20
NUM_WORKERS = 24

# Prevent Garbage Time: Set to full board size to ensure AI learns winning and losing.
MAX_MOVES = 225

# Learning Rate Strategy
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-4

BOARD_SIZE = 15
WIN_COUNT = 5
EPOCHS_PER_UPDATE = 5
C_PUCT = 2.0

MAX_ITERS = 10000
MODEL_DIR = "models_v3"
os.makedirs(MODEL_DIR, exist_ok=True)

device_train = torch.device("cuda")
device_selfplay = torch.device("cuda")


# ==================== Core Components ====================

class GomokuBoard:
    """Manages the Gomoku board state."""

    def __init__(self):
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        self.current_player = 1
        self.last_move = (-1, -1)

    def copy(self):
        """Creates a deep copy of the board."""
        new = GomokuBoard()
        new.board = self.board.copy()
        new.current_player = self.current_player
        new.last_move = self.last_move
        return new

    def get_state(self):
        """Returns the board state as a stacked numpy array for neural network input."""
        me = (self.board == self.current_player).astype(np.float32)
        opp = (self.board == 3 - self.current_player).astype(np.float32)
        color = np.full((BOARD_SIZE, BOARD_SIZE), self.current_player / 2.0, dtype=np.float32)

        last_move_map = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        lx, ly = self.last_move
        if lx >= 0 and ly >= 0:
            last_move_map[lx, ly] = 1.0

        return np.stack([me, opp, color, last_move_map])

    def move(self, x, y):
        """Places a move on the board if valid."""
        if not (0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE): return False
        if self.board[x, y] != 0: return False
        self.board[x, y] = self.current_player
        self.last_move = (x, y)
        self.current_player = 3 - self.current_player
        return True

    def check_win(self):
        """Checks for a winner or draw after the last move."""
        lx, ly = self.last_move
        if lx == -1: return 0
        player = self.board[lx, ly]
        if player == 0: return 0

        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for dx, dy in directions:
            count = 1
            for k in range(1, WIN_COUNT):
                nx, ny = lx + k * dx, ly + k * dy
                if 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE and self.board[nx, ny] == player:
                    count += 1
                else:
                    break
            for k in range(1, WIN_COUNT):
                nx, ny = lx - k * dx, ly - k * dy
                if 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE and self.board[nx, ny] == player:
                    count += 1
                else:
                    break
            if count >= WIN_COUNT: return player
        if np.all(self.board != 0): return 3
        return 0

    def get_legal_moves(self):
        """Returns a list of legal move indices, prioritizing moves near occupied positions."""
        if np.all(self.board == 0):
            center = BOARD_SIZE // 2
            return [center * BOARD_SIZE + center]

        indices = np.argwhere(self.board == 0)
        occupied = np.argwhere(self.board != 0)
        mask = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=bool)

        for ox, oy in occupied:
            r_min, r_max = max(0, ox - 2), min(BOARD_SIZE, ox + 3)
            c_min, c_max = max(0, oy - 2), min(BOARD_SIZE, oy + 3)
            mask[r_min:r_max, c_min:c_max] = True

        legal_moves = []
        for x, y in indices:
            if mask[x, y]:
                legal_moves.append(x * BOARD_SIZE + y)

        if not legal_moves:
            return [x * BOARD_SIZE + y for x, y in indices]

        return legal_moves


# ==================== Network Architecture (SE-ResNet) ====================

class SEResidualBlock(nn.Module):
    """Squeeze-and-Excitation Residual Block."""

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
    """AlphaZero-style neural network with policy and value heads."""

    def __init__(self, num_blocks=NUM_BLOCKS, channels=CHANNELS):
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
        p = F.relu(self.bn_p(self.conv_p(x))).view(x.size(0), -1)
        p = self.fc_p(p)
        p = F.softmax(p, dim=1)
        v = F.relu(self.bn_v(self.conv_v(x))).view(x.size(0), -1)
        v = F.relu(self.fc_v1(v))
        v = torch.tanh(self.fc_v2(v))
        return p, v


# ==================== MCTS ====================

class MCTS:
    """Monte Carlo Tree Search for action selection."""

    def __init__(self, net, simulations):
        self.net = net
        self.simulations = simulations
        self.Q = {}
        self.N = {}
        self.P = {}
        self.W = {}

    def clear(self):
        """Clears the tree statistics."""
        self.Q.clear();
        self.N.clear();
        self.P.clear();
        self.W.clear()

    def search(self, board: GomokuBoard, depth=0):
        """Performs a recursive MCTS search."""
        if depth > 256: return 0.0
        s = tuple(board.board.flatten())
        winner = board.check_win()
        if winner != 0: return 0.0 if winner == 3 else -1.0

        if s not in self.P:
            state = torch.from_numpy(board.get_state()).unsqueeze(0).to(device_selfplay, non_blocking=True)
            with torch.no_grad():
                p, v = self.net(state)
            p = p.cpu().numpy()[0]
            v = v.item()

            legal = board.get_legal_moves()
            full_mask = np.zeros(BOARD_SIZE * BOARD_SIZE)
            full_mask[legal] = 1
            p = p * full_mask

            if p.sum() > 0:
                p /= p.sum()
            else:
                p[legal] = 1.0 / len(legal)

            self.P[s] = p
            self.Q[s] = self.N[s] = self.W[s] = 0
            return -v

        best_u, best_a = -float('inf'), -1
        ns = math.sqrt(self.N[s] + 1)
        legal_moves = board.get_legal_moves()

        for a in legal_moves:
            u = self.Q.get((s, a), 0) + C_PUCT * self.P[s][a] * ns / (1 + self.N.get((s, a), 0))
            if u > best_u: best_u, best_a = u, a

        next_board = board.copy()
        next_board.move(best_a // BOARD_SIZE, best_a % BOARD_SIZE)
        v = self.search(next_board, depth + 1)
        sa = (s, best_a)
        self.N[sa] = self.N.get(sa, 0) + 1
        self.W[sa] = self.W.get(sa, 0) + v
        self.Q[sa] = self.W[sa] / self.N[sa]
        self.N[s] += 1
        return -v


_worker_net = None


def init_worker(shared_state_dict):
    """Initializes the worker with the shared network state."""
    global _worker_net
    _worker_net = AlphaZeroNet(num_blocks=NUM_BLOCKS, channels=CHANNELS).to(device_selfplay)
    _worker_net.load_state_dict(shared_state_dict)
    _worker_net.eval()


def self_play_worker(simulations):
    """Performs self-play games to generate training data."""
    global _worker_net
    board = GomokuBoard()
    mcts = MCTS(_worker_net, simulations + np.random.randint(-20, 20))
    states, pis, players = [], [], []

    # Temperature threshold: Explore in first 30 moves, then exploit.
    TEMPERATURE_MOVE_THRESHOLD = 30

    for step in range(MAX_MOVES):
        mcts.clear()
        for _ in range(mcts.simulations): mcts.search(board.copy())

        s_key = tuple(board.board.flatten())
        pi = np.zeros(BOARD_SIZE * BOARD_SIZE, dtype=np.float32)
        legal = board.get_legal_moves()
        for a in legal: pi[a] = mcts.N.get((s_key, a), 0)
        if pi.sum() > 0: pi /= pi.sum()

        players.append(board.current_player)
        states.append(board.get_state())
        pis.append(pi)

        action = 0
        if step < TEMPERATURE_MOVE_THRESHOLD:
            # Early game: Add Dirichlet noise for exploration.
            legal_probs = pi[legal]
            noise = np.random.dirichlet([DIRICHLET_ALPHA] * len(legal))
            legal_probs = 0.75 * legal_probs + 0.25 * noise
            if legal_probs.sum() > 0: legal_probs /= legal_probs.sum()
            action = legal[np.random.choice(len(legal), p=legal_probs)]
        else:
            # Late game: Select the best action deterministically.
            action = np.argmax(pi)

        board.move(action // BOARD_SIZE, action % BOARD_SIZE)

        winner = board.check_win()
        if winner != 0:
            values = []
            for p in players:
                if winner == 3:
                    values.append(0.0)
                else:
                    values.append(1.0 if winner == p else -1.0)
            return list(zip(states, pis, values))

    return list(zip(states, pis, [0.0] * len(states)))


class ReplayBuffer:
    """Stores experience tuples for training."""

    def __init__(self, capacity=REPLAY_BUFFER_SIZE):
        self.states = deque(maxlen=capacity)
        self.policies = deque(maxlen=capacity)
        self.values = deque(maxlen=capacity)

    def push(self, s, p, v):
        self.states.append(s);
        self.policies.append(p);
        self.values.append(v)

    def sample(self, batch_size):
        if len(self.states) < batch_size: return None
        idx = np.random.choice(len(self.states), batch_size, replace=False)
        return (np.array([self.states[i] for i in idx], dtype=np.float32),
                np.array([self.policies[i] for i in idx], dtype=np.float32),
                np.array([self.values[i] for i in idx], dtype=np.float32))

    def __len__(self): return len(self.states)


def apply_symmetry_batch(states, policies):
    """Applies random rotations and flips to augment the data batch."""
    B = states.shape[0]
    states = states.reshape(B, INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE)
    policies = policies.reshape(B, BOARD_SIZE, BOARD_SIZE)
    k = np.random.randint(0, 4)
    flip = np.random.randint(0, 2)
    states = np.rot90(states, k, axes=(2, 3))
    policies = np.rot90(policies, k, axes=(1, 2))
    if flip:
        states = np.flip(states, axis=3)
        policies = np.flip(policies, axis=2)
    return np.ascontiguousarray(states), np.ascontiguousarray(policies).reshape(B, -1)


def train_step(net, optimizer, scaler, buffer):
    """Performs a single training step on a batch from the buffer."""
    data = buffer.sample(BATCH_SIZE)
    if not data: return None
    s, p_t, v_t = data
    s, p_t = apply_symmetry_batch(s, p_t)
    s = torch.tensor(s, device=device_train)
    p_t = torch.tensor(p_t, device=device_train)
    v_t = torch.tensor(v_t, device=device_train).unsqueeze(1)

    net.train()
    optimizer.zero_grad()
    with autocast("cuda"):
        p, v = net(s)
        loss = -(p_t * torch.log(p + 1e-8)).sum(1).mean() + F.mse_loss(v, v_t)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    return loss.item()


def main():
    """Main training loop for the AlphaZero Gomoku AI."""
    mp.set_start_method("spawn", force=True)

    print(f" V3.2 Battle Ready | Device: {device_train}")
    print(f"Blocks: {NUM_BLOCKS} | SE-Block: On | LastMove: On | Pruning: On")
    print(f"Batch: {BATCH_SIZE} | Alpha: {DIRICHLET_ALPHA} | MaxMoves: {MAX_MOVES}")

    net = AlphaZeroNet(num_blocks=NUM_BLOCKS, channels=CHANNELS).to(device_train)
    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=MAX_ITERS, eta_min=1e-4
    )

    scaler = GradScaler("cuda")
    buffer = ReplayBuffer()

    start_iter = 1
    if os.path.exists("checkpoint_v3.pth"):
        print("Found checkpoint, loading...")
        try:
            cp = torch.load("checkpoint_v3.pth", weights_only=False)

            net.load_state_dict(cp['net'])
            optimizer.load_state_dict(cp['opt'])
            buffer.states = cp['buffer']['s']
            buffer.policies = cp['buffer']['p']
            buffer.values = cp['buffer']['v']
            start_iter = cp['iter'] + 1
            print(f" Resumed V3 Training from Iter {start_iter - 1}")

            # Reset LR
            for param_group in optimizer.param_groups:
                param_group['lr'] = LEARNING_RATE
            print(f" LR Reset to {LEARNING_RATE}")

        except Exception as e:
            print(f" Load Failed ({e}), Starting Fresh.")
    else:
        print(" Starting Fresh Training")

    try:
        for iteration in range(start_iter, MAX_ITERS + 1):
            sims = min(BASE_SIMULATIONS + (iteration // 5) * 10, MAX_SIMULATIONS)
            t0 = time.time()

            # Self-play
            net.eval()
            shared = {k: v.cpu() for k, v in net.state_dict().items()}
            with mp.Pool(NUM_WORKERS, initializer=init_worker, initargs=(shared,)) as pool:
                results = list(
                    tqdm(pool.imap_unordered(self_play_worker, [sims] * SELF_PLAY_GAMES), total=SELF_PLAY_GAMES,
                         desc=f"Iter {iteration} SP"))

            new_data = 0
            for res in results:
                new_data += len(res)
                for x in res: buffer.push(*x)

            # Training
            losses = []
            if len(buffer) >= MIN_BUFFER_TO_TRAIN:
                for _ in range(EPOCHS_PER_UPDATE):
                    l = train_step(net, optimizer, scaler, buffer)
                    if l: losses.append(l)
                scheduler.step()

            t1 = time.time()
            avg_loss = np.mean(losses) if losses else 0
            current_lr = optimizer.param_groups[0]['lr']
            print(
                f"Iter {iteration} | Loss: {avg_loss:.4f} | LR: {current_lr:.2e} | Data: +{new_data} | Time: {t1 - t0:.1f}s")

            if iteration % 20 == 0:
                torch.save({'net': net.state_dict(), 'opt': optimizer.state_dict(), 'iter': iteration,
                            'buffer': {'s': buffer.states, 'p': buffer.policies, 'v': buffer.values}},
                           "checkpoint_v3.pth")
                torch.save(net.state_dict(), os.path.join(MODEL_DIR, f"model_iter{iteration}.pth"))

    except KeyboardInterrupt:
        print("\n Saving V3 Checkpoint...")
        torch.save({'net': net.state_dict(), 'opt': optimizer.state_dict(), 'iter': iteration,
                    'buffer': {'s': buffer.states, 'p': buffer.policies, 'v': buffer.values}}, "checkpoint_v3.pth")


if __name__ == "__main__":
    main()