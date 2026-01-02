# Gomoku AlphaZero V3
A complete implementation of AlphaZero for Gomoku (15×15) with advanced features like Squeeze-and-Excitation Residual Networks, Monte Carlo Tree Search, and self-play training.
## Features

Advanced Neural Architecture: SE-ResNet with 10 blocks and 128 channels
Efficient MCTS: Optimized search with legal move pruning and positional priors
Self-Play Training: Parallelized data generation with Dirichlet noise exploration
Human vs AI Testing: Interactive Pygame interface with model selection
Training Resilience: Checkpoint saving, learning rate scheduling, and data augmentation
## Project Structure
├── test_model.py          # Standalone AI tester with GUI
├── train.py              # Training script with self-play
├── models_v3/            # Directory for trained models
├── checkpoint_v3.pth     # Training checkpoint (auto-generated)
└── README.md            # This file

## Requirements
Python 3.8+
PyTorch 2.0+
Pygame
NumPy
tqdm
Install dependencies:
pip install -r requirements.txt

## Quick Start

### Create virtual environment
python -m venv gomoku_env

### Activate virtual environment

### Windows:
gomoku_env\Scripts\activate

### Linux/Mac:
source gomoku_env/bin/activate

### Install all dependencies at once:

pip install -r requirements.txt

### 1. Play Against the AI
Run the tester to play against a trained model:

python test_model.py

Choose who moves first (Human or AI)

Select a model from the models_v3directory

Click on the board to place stones

Use Rto reset, ESCto quit

### 2. Train Your Own AI

Start training from scratch:

python train.py

The training process includes:
Self-play generation: 20 parallel games per iteration
Progressive difficulty: Simulations increase from 400 to 800
Automatic checkpointing: Saves every 20 iterations
Learning rate decay: Cosine annealing from 5e-4 to 1e-4

## Key Configuration

### Neural Network
Architecture: SE-ResNet with 10 residual blocks
Input channels: 4 (player, opponent, color, last move)
Output heads: Policy (225 moves) + Value (-1 to 1)
### Training Parameters
Batch size: 4096
MCTS simulations: 400-800 (increasing)
Replay buffer: 30,000 positions
Dirichlet noise: α=0.1 for exploration
Max moves: 225 (full board)
### Game Rules
Board size: 15×15
Win condition: 5 stones in a row
First move: Center position for AI
