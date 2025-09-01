# snake-rl-agent
Q-Learning snake agent with custom Gymnasium env

# Snake RL Agent 🐍 (Q-Learning)

This project implements a **Q-Learning agent** to play the classic Snake game.  
A custom **Gymnasium-compatible environment** was developed to simulate the game and train the agent.

## ✨ Features
- Custom Gym environment with 96 discrete states and 3 possible actions (left, straight, right).
- Reward shaping strategy (apple +10, collision −10, step penalty −0.02, distance-based shaping).
- Training, evaluation, and watching scripts:
  - `qlearn_snake.py` → Train the agent
  - `evaluate_snake.py` → Evaluate performance headlessly
  - `watch_snake.py` → Watch the trained agent in action
  - `play_snake.py` → Play the game manually

## 📦 Installation
```bash
python -m venv .venv
# activate your virtual environment
pip install -r requirements.txt

🚀 Usage

Train (headless) --> python qlearn_snake.py --episodes 10000 --save_path q_table.npy
Train with rendering every n episodes --> python qlearn_snake.py --episodes 3000 --render_every 300 --save_path q_table.npy
Continue training from checkpoint --> python qlearn_snake.py --episodes 5000 --load_path q_table.npy --save_path q_table.npy
Evaluate --> python evaluate_snake.py --episodes 200 --q_path q_table.npy
Watch trained agent --> python watch_snake.py --q_path q_table.npy --episodes 5 --fps 10
Play manually (WASD / arrow keys) --> python play_snake.py
