Gomoku AI (PPO Reinforcement Learning)
A complete Gomoku (Five in a Row) AI system based on the PPO (Proximal Policy Optimization) reinforcement learning algorithm, supporting both training mode and game mode.
Project Overview
This project implements a complete Gomoku AI system with the following features:
Algorithm Foundation: Based on PPO (Proximal Policy Optimization) algorithm
Network Architecture: Uses ResNet residual network for feature extraction
Training Optimization: Supports parallel environment training for improved efficiency
Game Interface: Pygame-based visual interface
Model Evaluation: Built-in model performance evaluation functionality
Environment Requirements
System Requirements
Python 3.7+
CUDA-supported GPU (Recommended, not required)
Create Virtual Environment
# Create virtual environment
python -m venv gomoku_env

# Activate virtual environment

# Windows:
gomoku_env\Scripts\activate

# Linux/Mac:
source gomoku_env/bin/activate

# Install all dependencies at once:

python.exe -m pip install --upgrade pip   

pip install -r requirements.txt

# Run

py play.py

# usage
1. Training Mode
Purpose: Train the AI model
Activation Method:
Edit the if __name__ == "__main__":section at the end of the code file:

if __name__ == "__main__":
    # Training Mode - uncomment the following line
    train()

    # Evaluation Mode - comment the following lines
    # evaluate("models-V2/agent_latest.pth", test_games=10)

    # UI Mode - comment the following lines
    # ui = GomokuUI("models-V2/agent_latest.pth")
    # ui.run()

2. Evaluation Mode
Purpose: Test AI performance against random agents
Activation Method:
if __name__ == "__main__":
    # Training Mode - comment the following line
    # train()

    # Evaluation Mode - uncomment the following line
    evaluate("models-V2/agent_latest.pth", test_games=10)

    # UI Mode - comment the following lines
    # ui = GomokuUI("models-V2/agent_latest.pth")
    # ui.run()

3. UI Interactive Mode (Test Mode)
Purpose: Visual interface for human vs AI gameplay
Activation Method:
if __name__ == "__main__":
    # Training Mode - comment the following line
    # train()

    # Evaluation Mode - comment the following line
    # evaluate("models-V2/agent_latest.pth", test_games=10)

    # UI Mode - uncomment the following lines
    ui = GomokuUI("models-V2/agent_latest.pth")
    ui.run()

# File Structure

.
├── play.py          # Main program file
├── models-V2/            # Model directory (auto-created)
│   └── agent_latest.pth  # Trained model
├── requirements.txt       # Dependencies file
└── README.md            # This documentation