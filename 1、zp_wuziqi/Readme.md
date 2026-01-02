# AlphaZero Gomoku (Five in a Row)

An implementation of AlphaZero algorithm for the game of Gomoku (Five in a Row) using PyTorch and Pygame.

## Overview

This project implements the AlphaZero reinforcement learning algorithm to play Gomoku, a classic board game also known as Five in a Row. The system includes:
Self-play training using Monte Carlo Tree Search (MCTS)
Deep neural network with residual blocks for policy and value estimation
Interactive GUI for human vs AI gameplay
Configurable training parameters for different board sizes and difficulty levels

# Features

🧠 AlphaZero algorithm with MCTS
🎮 Interactive Pygame interface
⚡ Mixed precision training support
🔧 Configurable board sizes and game rules
💾 Model checkpointing and resume capability
📊 Training diagnostics and evaluation
Prerequisites
Python 3.8+
PyTorch 2.0+
CUDA-capable GPU (recommended for training)

Installation

1. Create Virtual Environment
## Create virtual environment
python -m venv gomoku_env

## Activate virtual environment
## On Windows:

gomoku_env\Scripts\activate

## On macOS/Linux:

source gomoku_env/bin/activate

2. Install Dependencies

## Upgrade pip

python.exe -m pip install --upgrade pip

pip install -r requirements.txt

# Project Structure
gomoku-alphazero/
├── main.py              # Main entry point
├── config.py            # Configuration parameters
├── train.py             # Training script
├── test.py              # Testing/playing script
├── agent.py             # Neural network and MCTS
├── environment.py       # Game logic and board
├── gameUI.py            # Pygame interface
├── game.py              # Game state management
├── tools.py             # Utility functions
├── models/              # Directory for saved models
└── README.md
# Usage
## Training Mode
### To train the AI from scratch:

Edit config.py to set IS_TEST = False

python main.py


### To play against the trained AI:

Edit config.py to set IS_TEST = True

python main.py

## Game Controls
Mouse: Click to place stones

ESC: Exit the game


## Configuration
Key parameters in config.py:
BOARD_SIZE = 15           # Board dimensions
WIN_COUNT = 5             # Stones needed to win
NUM_WORKERS = 16          # Parallel self-play workers
LEARNING_RATE = 3e-4     # Neural network learning rate
MAX_ITERS = 600           # Training iterations
DEVICE_TRAIN = "cuda"     # Training device (cuda/cpu)

Training Process
The training follows these steps:
Self-play: AI plays against itself using MCTS
Data collection: Game positions and outcomes are stored
Network training: Neural network learns from collected data
Evaluation: Model performance is tested against baseline
Iteration: Process repeats with improved model