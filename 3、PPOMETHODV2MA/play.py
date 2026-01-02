import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.categorical import Categorical
import pygame
import os
import time
import sys
from gymnasium import spaces


class Config:
    # ------------------ Environmental Basic Configuration ------------------
    BOARD_SIZE = 15
    N_IN_ROW = 5
    OBS_CHANNELS = 3  # [Self, Opponent, LastMove]

    # ------------------ PPO & Hyperparameters ------------------
    LEARNING_RATE = 2.5e-4
    GAMMA = 0.99
    GAE_LAMBDA = 0.95
    CLIP_EPSILON = 0.2
    ENTROPY_COEF = 0.02
    VALUE_LOSS_COEF = 0.5
    MAX_GRAD_NORM = 0.5

    HIDDEN_CHANNELS = 128
    NUM_RES_BLOCKS = 8

    # -------------- Parallel Configuration ----------------
    NUM_ENVS = 24
    STEPS_PER_ENV = 512

    BATCH_SIZE = NUM_ENVS * STEPS_PER_ENV
    MINIBATCH_SIZE = 2048
    UPDATE_EPOCHS = 10

    # ------------------ Training Configuration ------------------
    TOTAL_TIMESTEPS = 5e7

    # ------------------ Device Configuration ------------------
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SEED = 42


class GomokuRules:
    """Game rules, also the reward model of PPO"""

    def __init__(self):
        self.size = Config.BOARD_SIZE
        self.n = Config.N_IN_ROW

    def is_legal_move(self, board, x, y):
        if x < 0 or x >= self.size or y < 0 or y >= self.size:
            return False
        return board[x][y] == 0

    def check_win(self, board, x, y, player):
        """Winning/losing detection"""
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for dx, dy in directions:
            count = 1
            # Positive direction
            for step in range(1, self.n + 1):
                nx, ny = x + step * dx, y + step * dy
                if 0 <= nx < self.size and 0 <= ny < self.size and board[nx][ny] == player:
                    count += 1
                else:
                    break

            # Negative direction
            for step in range(1, self.n + 1):
                nx, ny = x - step * dx, y - step * dy
                if 0 <= nx < self.size and 0 <= ny < self.size and board[nx][ny] == player:
                    count += 1
                else:
                    break

            if count >= self.n:
                return True
        return False

    def get_legal_moves_mask(self, board):
        """Get legal moves"""
        return (board.flatten() == 0)

    def _check_line(self, board, x, y, dx, dy, player):
        """Calculate the length and vacancy status of a chain passing through (x, y) with direction (dx, dy)"""
        length = 1
        open_ends = 0

        i = 1
        while True:
            nx, ny = x + i * dx, y + i * dy
            if 0 <= nx < self.size and 0 <= ny < self.size and board[nx][ny] == player:
                length += 1
                i += 1
            else:
                if 0 <= nx < self.size and 0 <= ny < self.size and board[nx][ny] == 0:
                    open_ends += 1
                break

        j = 1
        while True:
            nx, ny = x - j * dx, y - j * dy
            if 0 <= nx < self.size and 0 <= ny < self.size and board[nx][ny] == player:
                length += 1
                j += 1
            else:
                if 0 <= nx < self.size and 0 <= ny < self.size and board[nx][ny] == 0:
                    open_ends += 1
                break

        return length, open_ends

    def _evaluate_point(self, board, x, y, player):
        """Evaluate the reward of a point on the board"""
        scores = 0.0
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

        # Reward Table
        SCORE_LIVE_4 = 0.80
        SCORE_SLEEP_4 = 0.30
        SCORE_LIVE_3 = 0.20
        SCORE_SLEEP_3 = 0.05
        SCORE_LIVE_2 = 0.02

        for dx, dy in directions:
            length, open_ends = self._check_line(board, x, y, dx, dy, player)

            if length >= 5:
                return 2.0
            elif length == 4:
                if open_ends == 2:
                    scores += SCORE_LIVE_4
                elif open_ends == 1:
                    scores += SCORE_SLEEP_4
            elif length == 3:
                if open_ends == 2:
                    scores += SCORE_LIVE_3
                elif open_ends == 1:
                    scores += SCORE_SLEEP_3
            elif length == 2:
                if open_ends == 2:
                    scores += SCORE_LIVE_2

        return scores

    def calculate_heuristic_reward(self, board, x, y, current_player):
        """Balance the reward for attack and defense"""
        opponent = -current_player
        attack_score = self._evaluate_point(board, x, y, current_player)
        defense_score = self._evaluate_point(board, x, y, opponent)
        defense_weight = 1.2
        total_reward = attack_score + (defense_score * defense_weight)
        return total_reward


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Layer initialization"""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class ResidualBlock(nn.Module):
    """Residual Block: The core component of ResNet backbone."""

    def __init__(self, channels):
        super().__init__()
        self.conv1 = layer_init(nn.Conv2d(channels, channels, 3, padding=1))
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = layer_init(nn.Conv2d(channels, channels, 3, padding=1))
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        return F.relu(x + self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x))))))


class ActorCritic(nn.Module):
    """Model for PPO with ResNet backbone and two heads"""

    def __init__(self):
        super().__init__()
        self.board_width = Config.BOARD_SIZE
        h_dim = Config.HIDDEN_CHANNELS

        # Feature Extraction
        self.conv_in = layer_init(nn.Conv2d(Config.OBS_CHANNELS, h_dim, 3, padding=1))

        # ResNet Backbone
        blocks = []
        for _ in range(Config.NUM_RES_BLOCKS):
            blocks.append(ResidualBlock(h_dim))
        self.res_blocks = nn.Sequential(*blocks)

        # Actor Head
        self.actor = nn.Sequential(
            layer_init(nn.Conv2d(h_dim, 32, 1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(32 * self.board_width ** 2, self.board_width ** 2), std=0.01)
        )

        # Critic Head
        self.critic = nn.Sequential(
            layer_init(nn.Conv2d(h_dim, 3, 1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(3 * self.board_width ** 2, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 1), std=1)
        )

    def get_value(self, x):
        """Get value function approximation"""
        x = F.relu(self.conv_in(x))
        x = self.res_blocks(x)
        return self.critic(x)

    def get_action_and_value(self, x, action=None, action_mask=None, deterministic=False):
        """Get both policy and value function approximations"""
        x = F.relu(self.conv_in(x))
        x = self.res_blocks(x)

        logits = self.actor(x)

        if action_mask is not None:
            logits = torch.where(action_mask, logits, torch.tensor(-1e8).to(x.device))

        probs = Categorical(logits=logits)

        if action is None:
            if deterministic:
                action = torch.argmax(logits, dim=1)
            else:
                action = probs.sample()

        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


class GomokuEnv(gym.Env):
    """Gomoku Environment for Reinforcement Learning"""

    def __init__(self):
        super().__init__()
        self.size = Config.BOARD_SIZE
        self.rules = GomokuRules()

        self.action_space = spaces.Discrete(self.size * self.size)
        self.observation_space = spaces.Box(
            low=0, high=1,
            shape=(Config.OBS_CHANNELS, self.size, self.size),
            dtype=np.float32
        )

        self.board = None
        self.current_player = 1
        self.last_move = None
        self.steps = 0

    def reset(self):
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.current_player = 1
        self.last_move = None
        self.steps = 0
        return self._get_obs(), self._get_info()

    def step(self, action):
        x, y = action // self.size, action % self.size

        # Penalty for illegal moves
        if not self.rules.is_legal_move(self.board, x, y):
            return self._get_obs(), -1.0, True, False, {"error": "Invalid Move"}

        # Execute move
        self.board[x][y] = self.current_player
        self.last_move = (x, y)
        self.steps += 1

        # Calculate rewards and check if game is done
        won = self.rules.check_win(self.board, x, y, self.current_player)
        done = won or (self.steps >= self.size * self.size)

        reward = 0.0

        if won:
            reward = 10.0
        elif done:
            reward = 0.0  # Draw
        else:
            shape_reward = self.rules.calculate_heuristic_reward(self.board, x, y, self.current_player)
            time_penalty = -0.005
            reward = shape_reward + time_penalty

        info = self._get_info()
        self.current_player *= -1

        return self._get_obs(), reward, done, False, info

    def _get_obs(self):
        obs = np.zeros((Config.OBS_CHANNELS, self.size, self.size), dtype=np.float32)

        me = self.current_player
        opponent = -self.current_player

        obs[0] = (self.board == me).astype(float)
        obs[1] = (self.board == opponent).astype(float)

        if self.last_move:
            obs[2][self.last_move] = 1.0

        return obs

    def _get_info(self):
        return {"action_mask": self.rules.get_legal_moves_mask(self.board)}


class VectorizedEnv:
    """Vectorized environment that runs multiple games in parallel."""

    def __init__(self, num_envs):
        self.envs = [GomokuEnv() for _ in range(num_envs)]
        self.num_envs = num_envs

    def reset(self):
        obs_list = []
        info_list = []
        for env in self.envs:
            o, i = env.reset()
            obs_list.append(o)
            info_list.append(i)
        return np.array(obs_list), info_list

    def step(self, actions):
        results = []
        for i in range(self.num_envs):
            o, r, term, trunc, info = self.envs[i].step(actions[i])
            if term or trunc:
                o, info = self.envs[i].reset()
            results.append((o, r, term, trunc, info))
        obs, rews, terms, truncs, infos = zip(*results)
        return np.array(obs), np.array(rews), np.array(terms), np.array(truncs), list(infos)


def random_agent(env):
    """Random opponent: randomly selects from legal moves"""
    mask = env.rules.get_legal_moves_mask(env.board)
    legal_indices = np.flatnonzero(mask)

    if len(legal_indices) == 0:
        return None

    return np.random.choice(legal_indices)


def evaluate(model_path="models-V1/agent_latest-temp.pth", test_games=20):
    """Evaluate the trained model against random agent"""
    print(f"{'=' * 40}")
    print(f"Loading model from: {model_path}")
    print(f"{'=' * 40}")

    # Load model
    try:
        agent = ActorCritic().to(Config.DEVICE)
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=Config.DEVICE)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                agent.load_state_dict(checkpoint['model_state_dict'])
            else:
                agent.load_state_dict(checkpoint)
            print("Model loaded successfully.")
        else:
            print(f"Error: Model file not found at {model_path}")
            return
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    agent.eval()

    ai_wins = 0
    draws = 0
    losses = 0

    print(f"\nStarting Evaluation: PPO Agent vs Random Agent ({test_games} games)")
    print("-" * 60)

    for i in range(test_games):
        env = GomokuEnv()
        obs_np, info = env.reset()

        ai_player = 1 if (i % 2 == 0) else -1
        role_str = "Black (First)" if ai_player == 1 else "White (Second)"
        print(f"Game {i + 1}/{test_games} | AI is {role_str}", end=" ... ")

        done = False
        game_steps = 0

        while not done:
            is_ai_turn = (env.current_player == ai_player)

            if is_ai_turn:
                obs_tensor = torch.tensor(obs_np, dtype=torch.float32).unsqueeze(0).to(Config.DEVICE)
                mask_tensor = torch.tensor(info["action_mask"], dtype=torch.bool).unsqueeze(0).to(Config.DEVICE)

                with torch.no_grad():
                    action_idx, _, _, _ = agent.get_action_and_value(
                        obs_tensor, action_mask=mask_tensor, deterministic=True
                    )
                action = action_idx.item()
            else:
                action = random_agent(env)

            obs_np, reward, done, truncated, info = env.step(action)
            game_steps += 1

            if done:
                res = ""
                if reward > 1.0:
                    if is_ai_turn:
                        ai_wins += 1
                        res = "AI WIN"
                    else:
                        losses += 1
                        res = "AI LOSS"
                else:
                    draws += 1
                    res = "DRAW"

                print(f"{res} (Steps: {game_steps})")

    # Output evaluation report
    print("\n" + "=" * 40)
    print(" PERFORMANCE MEASUREMENTS REPORT")
    print("=" * 40)
    print(f"Model: {model_path}")
    print(f"Opponent: Random Agent Baseline")
    print(f"Total Games: {test_games}")
    print("-" * 40)
    print(f"AI Wins   : {ai_wins} ({ai_wins / test_games * 100:.1f}%)")
    print(f"AI Losses : {losses} ({losses / test_games * 100:.1f}%)")
    print(f"Draws     : {draws} ({draws / test_games * 100:.1f}%)")
    print(f"Accuracy  : {ai_wins / test_games:.2f}")
    print("=" * 40)


def train():
    """Main training function for PPO algorithm"""
    if not os.path.exists("models-V2"):
        os.makedirs("models-V2")

    # Initialize environment and agent
    envs = VectorizedEnv(Config.NUM_ENVS)
    agent = ActorCritic().to(Config.DEVICE)
    optimizer = optim.Adam(agent.parameters(), lr=Config.LEARNING_RATE, eps=1e-5)

    num_updates = int(Config.TOTAL_TIMESTEPS // Config.BATCH_SIZE)
    lr_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=num_updates)

    opponent = ActorCritic().to(Config.DEVICE)
    opponent.load_state_dict(agent.state_dict())

    for param in opponent.parameters():
        param.requires_grad = False

    # Initialize storage buffers
    obs = torch.zeros(
        (Config.STEPS_PER_ENV, Config.NUM_ENVS, Config.OBS_CHANNELS, Config.BOARD_SIZE, Config.BOARD_SIZE)).to(
        Config.DEVICE)
    actions = torch.zeros((Config.STEPS_PER_ENV, Config.NUM_ENVS)).to(Config.DEVICE)
    logprobs = torch.zeros((Config.STEPS_PER_ENV, Config.NUM_ENVS)).to(Config.DEVICE)
    rewards = torch.zeros((Config.STEPS_PER_ENV, Config.NUM_ENVS)).to(Config.DEVICE)
    dones = torch.zeros((Config.STEPS_PER_ENV, Config.NUM_ENVS)).to(Config.DEVICE)
    values = torch.zeros((Config.STEPS_PER_ENV, Config.NUM_ENVS)).to(Config.DEVICE)
    masks = torch.zeros((Config.STEPS_PER_ENV, Config.NUM_ENVS, Config.BOARD_SIZE ** 2), dtype=torch.bool).to(
        Config.DEVICE)

    global_step = 0
    next_obs_np, infos = envs.reset()
    next_obs = torch.Tensor(next_obs_np).to(Config.DEVICE)
    next_done = torch.zeros(Config.NUM_ENVS).to(Config.DEVICE)

    print(f"Hardware: {Config.DEVICE}, Envs: {Config.NUM_ENVS}, Batch: {Config.BATCH_SIZE}")
    print(f"Network: {Config.HIDDEN_CHANNELS} Channels, {Config.NUM_RES_BLOCKS} Blocks")
    print(f"Starting training for {num_updates} updates...")

    for update in range(1, num_updates + 1):
        # ---------------- Phase 1: Rollout ----------------
        for step in range(0, Config.STEPS_PER_ENV):
            global_step += Config.NUM_ENVS
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action_masks = torch.tensor(np.array([i["action_mask"] for i in infos])).to(Config.DEVICE)
                masks[step] = action_masks
                action, logprob, _, value = agent.get_action_and_value(next_obs, action_mask=action_masks)
                values[step] = value.flatten()

            actions[step] = action
            logprobs[step] = logprob

            real_action_np = action.cpu().numpy()
            next_obs_np, reward_np, term_np, trunc_np, infos = envs.step(real_action_np)

            done_np = np.logical_or(term_np, trunc_np)
            final_rewards = reward_np.copy().astype(float)

            # --- Opponent Move ---
            opp_indices = [i for i, d in enumerate(done_np) if not d]
            if len(opp_indices) > 0:
                opp_obs = torch.Tensor(next_obs_np[opp_indices]).to(Config.DEVICE)
                opp_masks = torch.tensor(np.array([infos[i]["action_mask"] for i in opp_indices])).to(Config.DEVICE)

                with torch.no_grad():
                    opp_action, _, _, _ = opponent.get_action_and_value(opp_obs, action_mask=opp_masks)

                for idx, env_idx in enumerate(opp_indices):
                    act = opp_action[idx].item()
                    o, r, term, trunc, info = envs.envs[env_idx].step(act)
                    if term or trunc:
                        o, info = envs.envs[env_idx].reset()
                        if r > 5.0:
                            final_rewards[env_idx] = -10.0
                        else:
                            final_rewards[env_idx] = 0.0
                        done_np[env_idx] = True
                    next_obs_np[env_idx] = o
                    infos[env_idx] = info

            rewards[step] = torch.tensor(final_rewards).to(Config.DEVICE).view(-1)
            next_obs = torch.Tensor(next_obs_np).to(Config.DEVICE)
            next_done = torch.Tensor(done_np).to(Config.DEVICE)

        # ---------------- Phase 2: GAE & Update ----------------
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(Config.DEVICE)
            lastgaelam = 0
            for t in reversed(range(Config.STEPS_PER_ENV)):
                if t == Config.STEPS_PER_ENV - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value.flatten()
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + Config.GAMMA * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + Config.GAMMA * Config.GAE_LAMBDA * nextnonterminal * lastgaelam
            returns = advantages + values

        b_obs = obs.reshape((-1,) + obs.shape[2:])
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_masks = masks.reshape((-1, Config.BOARD_SIZE ** 2))

        inds = np.arange(Config.BATCH_SIZE)
        for epoch in range(Config.UPDATE_EPOCHS):
            np.random.shuffle(inds)
            for start in range(0, Config.BATCH_SIZE, Config.MINIBATCH_SIZE):
                end = start + Config.MINIBATCH_SIZE
                mb_inds = inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], action=b_actions[mb_inds], action_mask=b_masks[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                mb_advantages = b_advantages[mb_inds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - Config.CLIP_EPSILON, 1 + Config.CLIP_EPSILON)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
                entropy_loss = entropy.mean()

                # Total loss
                loss = pg_loss - Config.ENTROPY_COEF * entropy_loss + Config.VALUE_LOSS_COEF * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), Config.MAX_GRAD_NORM)
                optimizer.step()

        lr_scheduler.step()

        # ---------------- Phase 3: Logging & Opponent Update ----------------
        if update % 5 == 0:
            print(
                f"Step {global_step}: P_Loss={pg_loss.item():.4f}, V_Loss={v_loss.item():.4f}, Rew={rewards.mean().item():.4f}, LR={lr_scheduler.get_last_lr()[0]:.6f}")

        # Update opponent model
        if update % 20 == 0:
            opponent.load_state_dict(agent.state_dict())
            torch.save(agent.state_dict(), f"models-V2/agent_latest.pth")
            print(f"==> [Update {update}] Updated Opponent and Saved Model.")


# UI Configuration
BG_COLOR = (230, 190, 140)
BLACK_COLOR = (0, 0, 0)
WHITE_COLOR = (255, 255, 255)
GRID_SIZE = 40
MARGIN = 40


class GomokuUI:
    """PyGame UI for playing against the trained AI"""

    def __init__(self, model_path=None):
        pygame.init()
        self.size = Config.BOARD_SIZE
        self.width = self.size * GRID_SIZE + 2 * MARGIN
        self.height = self.width
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("PPO Gomoku (AI First - Black)")

        self.board = np.zeros((self.size, self.size), dtype=int)
        self.rules = GomokuRules()
        self.game_over = False

        self.agent = ActorCritic().to(Config.DEVICE)
        if model_path:
            try:
                self.agent.load_state_dict(torch.load(model_path, map_location=Config.DEVICE))
                print(f"Loaded model: {model_path}")
            except:
                print("Model not found, using random weights (expect bad moves).")
        self.agent.eval()

    def draw_board(self):
        self.screen.fill(BG_COLOR)
        for i in range(self.size):
            start = MARGIN + i * GRID_SIZE
            end = MARGIN + (self.size - 1) * GRID_SIZE
            pygame.draw.line(self.screen, BLACK_COLOR, (start, MARGIN), (start, end), 2)
            pygame.draw.line(self.screen, BLACK_COLOR, (MARGIN, start), (end, start), 2)

        center = self.size // 2
        star_points = [center, 3, self.size - 4]
        for x in star_points:
            for y in star_points:
                pos = (MARGIN + x * GRID_SIZE, MARGIN + y * GRID_SIZE)
                pygame.draw.circle(self.screen, BLACK_COLOR, pos, 4)

    def draw_stones(self):
        for x in range(self.size):
            for y in range(self.size):
                if self.board[x][y] != 0:
                    center = (MARGIN + y * GRID_SIZE, MARGIN + x * GRID_SIZE)
                    color = BLACK_COLOR if self.board[x][y] == 1 else WHITE_COLOR
                    pygame.draw.circle(self.screen, color, center, GRID_SIZE // 2 - 2)

    def get_ai_move(self, player_val):
        obs = np.zeros((Config.OBS_CHANNELS, self.size, self.size), dtype=np.float32)
        obs[0] = (self.board == player_val).astype(float)
        obs[1] = (self.board == -player_val).astype(float)

        obs_tensor = torch.tensor(obs).unsqueeze(0).to(Config.DEVICE)
        mask = torch.tensor(self.rules.get_legal_moves_mask(self.board)).unsqueeze(0).to(Config.DEVICE)

        with torch.no_grad():
            action, _, _, _ = self.agent.get_action_and_value(obs_tensor, action_mask=mask, deterministic=True)

        return action.item() // self.size, action.item() % self.size

    def run(self):
        clock = pygame.time.Clock()
        running = True

        AI_PIECE = 1
        HUMAN_PIECE = -1

        current_player = 1

        print("Start! AI is Black (1) and goes First.")

        while running:
            self.draw_board()
            self.draw_stones()
            pygame.display.flip()

            move_made = False

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.MOUSEBUTTONUP and not self.game_over and current_player == HUMAN_PIECE:
                    if event.button == 1:
                        mx, my = event.pos
                        c = round((mx - MARGIN) / GRID_SIZE)
                        r = round((my - MARGIN) / GRID_SIZE)

                        if 0 <= r < self.size and 0 <= c < self.size:
                            if self.rules.is_legal_move(self.board, r, c):
                                self.board[r][c] = HUMAN_PIECE
                                move_made = True
                                print(f"Player (White): ({r}, {c})")
                                if self.rules.check_win(self.board, r, c, HUMAN_PIECE):
                                    print("Human Wins!")
                                    self.game_over = True

            if self.game_over:
                clock.tick(30)
                continue

            if move_made:
                current_player = AI_PIECE

            if current_player == AI_PIECE and not self.game_over:
                self.draw_board()
                self.draw_stones()
                pygame.display.flip()

                pygame.time.wait(200)

                r, c = self.get_ai_move(AI_PIECE)
                print(f"AI (Black): ({r}, {c})")

                self.board[r][c] = AI_PIECE

                if self.rules.check_win(self.board, r, c, AI_PIECE):
                    print("AI Wins!")
                    self.game_over = True

                current_player = HUMAN_PIECE

            clock.tick(30)
        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    # Example usage - uncomment the function you want to run

    # Train the model
    #train()

    # Evaluate the model
    # evaluate("models-V2/agent_latest.pth", test_games=10)

    # Run the UI to play against the AI
    ui = GomokuUI("models-V2/agent_latest.pth")
    ui.run()