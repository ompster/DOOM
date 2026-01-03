# VizDoom AI with Stable Baselines 3

Train AI agents to play DOOM using VizDoom and Proximal Policy Optimization (PPO) from Stable Baselines 3.

## System Specifications

This project was developed and tested on:
- **GPU**: NVIDIA RTX 3090 (24GB VRAM)
- **OS**: Windows 10/11 (win32)
- **Python**: 3.10+
- **RAM**: 16GB+ recommended
- **CUDA**: 11.8+ (for GPU acceleration)

## Overview

This project trains reinforcement learning agents to play various DOOM scenarios using:
- **VizDoom**: DOOM environment for RL research
- **Stable Baselines 3**: State-of-the-art RL algorithms (PPO)
- **PyTorch**: Deep learning framework with CUDA support
- **Gymnasium**: Modern RL environment API

The architecture features:
- Parallel training with 16 environments (optimized for RTX 3090)
- CNN-based visual processing (84x84x4 frame stacks)
- Comprehensive monitoring with TensorBoard
- Automatic checkpoint management
- Fast training with optimized hyperparameters

## Project Structure

```
DOOM/
├── src/
│   ├── envs/              # Environment wrappers
│   │   └── vizdoom_env.py # Main VizDoom wrapper with preprocessing
│   ├── callbacks/         # Training callbacks
│   │   ├── checkpoint_callback.py
│   │   ├── tensorboard_callback.py
│   │   └── eval_callback.py
│   └── utils/             # Utility functions
│
├── scripts/               # Training and evaluation scripts
│   ├── train_basic.py                      # Train on BASIC scenario
│   ├── train_basic_fast.py                 # Faster BASIC training
│   ├── train_defend_the_center.py          # Defend the Center scenario
│   ├── train_defend_center_gpu_max.py      # GPU-optimized version
│   ├── resume_defend_center_fast.py        # Resume with faster settings
│   ├── train_deadly_corridor_fast.py       # Deadly Corridor (fast)
│   ├── train_deadly_corridor_render.py     # Deadly Corridor (watch mode)
│   ├── train_deadly_corridor_harsh.py      # With reward shaping
│   ├── train_take_cover_fast.py            # Take Cover (fast)
│   ├── train_take_cover_render.py          # Take Cover (watch mode)
│   └── evaluate.py                         # Evaluate trained models
│
├── checkpoints/           # Model checkpoints (created during training)
├── logs/                  # TensorBoard logs (created during training)
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Setup

### 1. Create Virtual Environment

```bash
cd C:\Users\natha\Documents\Python\DOOM
python -m venv venv
```

### 2. Activate Virtual Environment

**Windows:**
```bash
venv\Scripts\activate
```

**Linux/Mac:**
```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Note for Windows:** VizDoom should install automatically via pip. If you encounter build issues, install Visual C++ Build Tools from Microsoft.

### 4. Verify Installation

Test that VizDoom is installed correctly:

```python
python -c "import vizdoom; import gymnasium; import torch; print('All dependencies installed!')"
```

Check CUDA availability:
```python
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Quick Start

### Easiest Scenario - Take Cover

Start with the simplest scenario (dodge fireballs by moving left/right):

```bash
# Fast training (no rendering)
python scripts/train_take_cover_fast.py

# Watch mode (all 8 envs visible)
python scripts/train_take_cover_render.py
```

**Expected results:**
- Starting: ~50-100 reward (random dodging)
- After 500k steps: ~500-1000 reward (learning patterns)
- After 2M steps: ~1500+ reward (mastery)

### Monitor Training with TensorBoard

In a new terminal (keep virtual env activated):

```bash
tensorboard --logdir=logs --port=6006
```

Then open http://localhost:6006 in your browser to view:
- Episode rewards and lengths
- Training loss curves
- Learning progress
- GPU utilization

Watch for the **rollout/ep_rew_mean** metric to track performance.

## Training Scenarios

### 1. Take Cover (EASIEST - RECOMMENDED START)

**Objective:** Dodge fireballs by strafing left/right
**Actions:** 2 (move left, move right)
**Reward:** +1 per step alive
**Difficulty:** Easy - should learn in 1-2M steps

```bash
# Fast training
python scripts/train_take_cover_fast.py

# Watch training
python scripts/train_take_cover_render.py
```

**Performance Metrics:**
- Good: 500-1000 reward (survives ~half episode)
- Expert: 1500-2000 reward (survives most of episode)
- Max: 2100 reward (survives full episode)

### 2. Defend The Center

**Objective:** Defend position, kill enemies, manage ammo
**Actions:** 3 (turn left, turn right, attack)
**Reward:** +1 per kill, -1 on death
**Difficulty:** Medium

```bash
# Original training
python scripts/train_defend_the_center.py

# GPU-optimized training
python scripts/train_defend_center_gpu_max.py

# Resume from checkpoint with faster settings
python scripts/resume_defend_center_fast.py
```

**Performance Metrics:**
- Starting: ~2-5 reward (random shooting)
- Good: 20-30 reward (learned aiming and tracking)
- Expert: 60-80 reward (efficient combat, good survival)
- Our result: **~22-24 reward** (plateaued - see lessons learned)

**Training time:** ~2 hours per 1M steps on RTX 3090

### 3. Basic Scenario

**Objective:** Shoot a single monster
**Actions:** 3 (move left, move right, attack)
**Reward:** -1 per step (encourages speed)
**Difficulty:** Easy

```bash
python scripts/train_basic.py
python scripts/train_basic_fast.py
```

### 4. Deadly Corridor (HARD - NOT RECOMMENDED)

**Objective:** Navigate corridor under heavy fire
**Actions:** 7 (full movement + turning + shooting)
**Reward:** Progress forward, -100 on death
**Difficulty:** Very Hard - reward hacking issues

```bash
# Standard training
python scripts/train_deadly_corridor_fast.py

# With harsh reward shaping
python scripts/train_deadly_corridor_harsh.py
```

**WARNING:** This scenario has severe reward hacking issues. See "Lessons Learned" section.

## Training Configuration

All training scripts use optimized hyperparameters for RTX 3090:

```python
N_ENVS = 16              # Parallel environments (optimized for memory)
BATCH_SIZE = 512         # GPU batch size
N_STEPS = 256           # Rollout buffer size
N_EPOCHS = 4            # Update epochs (reduced for speed)
LEARNING_RATE = 3e-4    # Learning rate
GAMMA = 0.99            # Discount factor
```

### Memory Considerations

- **16 envs + batch 512** requires ~10-12GB RAM + 2-4GB VRAM
- Original attempt with **48 envs** caused paging file errors on Windows
- Reduced to 16 envs for stability

### Training Speed

On RTX 3090:
- **Initial config**: ~8,000 steps/minute
- **Optimized config**: ~15-16,000 steps/minute
- **Bottleneck**: CPU/VizDoom overhead, not GPU (only 7% GPU utilization)

**Optimization strategies applied:**
- Reduced N_EPOCHS from 10 → 4 (2.5x speedup)
- Removed evaluation callback during training
- Less frequent checkpoints (every 500k steps)

## Evaluation

Evaluate a trained model:

```bash
# Watch the agent play (10 episodes)
python scripts/evaluate.py checkpoints/take_cover/final_take_cover.zip

# Specific checkpoint
python scripts/evaluate.py checkpoints/defend_the_center/ppo_defend_the_center_1000000_steps.zip

# 20 episodes without rendering (faster)
python scripts/evaluate.py checkpoints/take_cover/best_model.zip -n 20 --no-render

# Use stochastic actions (more variety)
python scripts/evaluate.py checkpoints/take_cover/final_take_cover.zip --no-deterministic
```

## Lessons Learned - Our Training Journey

### Scenario 1: Defend The Center ✅ (Partial Success)

**Result:** Agent learned basics but plateaued at 22-24 reward

**Timeline:**
- 0-1M steps: 2.9 → 20 reward (learned to aim and shoot)
- 1M-2M steps: 20 → 23 reward (slow improvement)
- 2M+ steps: Plateaued at 22-24 reward (stopped improving)

**What Worked:**
- Agent learned to aim at enemies
- Consistent performance (low variance)
- Basic combat mechanics mastered

**What Didn't Work:**
- Got stuck in local optimum (spin-and-shoot strategy)
- Couldn't break through to 60-80 reward target
- Likely needed different hyperparameters or reward shaping

**Lessons:**
- Simple scenarios can have local optima
- Plateau detection is important
- Consider curriculum learning or reward shaping for complex tasks

### Scenario 2: Deadly Corridor ❌ (Failed - Reward Hacking)

**Result:** Severe reward hacking - agent learned to rush forward and die

**The Problem:**
- Agent gets ~240 reward in 14 steps by rushing forward
- Death penalty (-100) was overwhelmed by progress rewards
- Mean reward: +142 despite dying every episode
- Episodes lasted only 14 steps (should be up to 2100)

**Attempts to Fix:**

1. **Reward Shaping v1:** Scaled death penalty with survival time
   - Death penalty: -100 to -200 based on survival
   - **Failed**: Still profitable to rush

2. **Harsh Reward Shaping:** Massive penalties + scaled down rewards
   - All rewards scaled to 10%
   - Death penalty: -500 to -2000
   - **Failed**: Mean reward stayed at -500, no learning after 2M steps

**Root Cause:**
- Deadly corridor gives massive progress rewards for forward movement
- 7 actions + navigation + combat = too complex
- Reward structure fundamentally broken for RL without extensive shaping

**Lessons:**
- Not all scenarios have good reward structures
- Reward hacking is a critical failure mode
- Sometimes it's better to switch scenarios than fight bad rewards
- Simpler scenarios often learn better

### Scenario 3: Take Cover ✅ (SUCCESS!)

**Result:** Clean learning, steady improvement, no exploits

**Why It Works:**
- Only 2 actions (move left/right) - simple action space
- Clear reward: +1 per step alive - no confusing signals
- Simple objective: dodge fireballs - easy to understand
- No reward hacking exploits

**Performance:**
- Starting: ~50-100 reward
- Current: ~230 reward (improving steadily)
- Expected: 1500-2000 reward at convergence

**Lessons:**
- Start with simpler scenarios
- Clear reward structures are crucial
- Fewer actions = faster learning
- Take Cover is the best starting point for learning VizDoom RL

## Performance Expectations

### Training Times (RTX 3090)

| Timesteps | Take Cover | Defend The Center | Notes |
|-----------|------------|-------------------|-------|
| 250k | ~15 min | ~30 min | Initial learning |
| 500k | ~30 min | ~60 min | Basic competence |
| 1M | ~60 min | ~2 hours | Good performance |
| 5M | ~5 hours | ~10 hours | Near mastery |

### GPU Utilization

- **Expected**: 60-80% GPU utilization
- **Actual**: 7-15% GPU utilization
- **Bottleneck**: CPU/VizDoom preprocessing and inter-process communication
- **Solution**: More parallel environments would help, but limited by RAM

### Checkpoint Files

Checkpoints are saved with statistics in JSON format:

```json
{
  "timesteps": 500000,
  "n_calls": 31250,
  "n_episodes": 100,
  "mean_reward": 230.5,
  "std_reward": 45.2,
  "min_reward": 120.0,
  "max_reward": 580.0,
  "mean_ep_length": 230.5
}
```

Monitor `mean_reward` and `mean_ep_length` to track learning progress.

## Troubleshooting

### Paging File Too Small (Windows)

**Problem:** `Out of memory` errors with many parallel environments

**Symptoms:**
- Errors when creating 48 parallel environments
- System freezes or crashes during training

**Solution:**
```
1. Open System Properties → Advanced → Performance Settings
2. Go to Advanced tab → Virtual Memory → Change
3. Uncheck "Automatically manage paging file size"
4. Set custom size:
   - Initial size: 16384 MB (16 GB)
   - Maximum size: 32768 MB (32 GB)
5. Click Set → OK → Restart
```

**Our fix:** Reduced from 48 envs → 16 envs to avoid the issue entirely

### Training Speed Too Slow

**Problem:** Only ~8,000 steps/minute (expected ~20-30k)

**Solutions applied:**
- Reduced `N_EPOCHS` from 10 → 4 (2.5x speedup)
- Removed evaluation callback (removes overhead)
- Less frequent checkpoints (500k vs 200k)
- Result: ~15-16k steps/minute

### VizDoom Installation Issues (Windows)

**Problem:** Build errors during `pip install vizdoom`

**Solution:** Install Visual C++ 2015-2022 Redistributable from Microsoft

### Multiprocessing Errors (Windows)

**Problem:** `RuntimeError: An attempt has been made to start a new process...`

**Solution:** Ensure training script has `if __name__ == "__main__":` wrapper (all provided scripts include this)

### Agent Not Learning / Plateau

**Solutions:**
1. Check TensorBoard - is reward improving at all?
2. Verify environment works: test with random agent first
3. Try different scenario (Take Cover is easiest)
4. Increase entropy coefficient for more exploration
5. Consider reward shaping for complex scenarios
6. Train longer - some scenarios need 5-10M steps

### CUDA Out of Memory

**Solutions:**
- Reduce `N_ENVS` to 8
- Reduce `BATCH_SIZE` to 256
- Use CPU training: `device="cpu"` (much slower)

## Tips for Success

1. **Start with Take Cover**: Simplest scenario, clean rewards, fast learning
2. **Monitor Early**: Check TensorBoard after 250k steps to verify learning
3. **Watch Episode Length**: If episodes are very short (<50 steps), check for reward hacking
4. **Be Patient**: Even simple scenarios take 1-2M steps to learn
5. **Use GPU Monitoring**: Run `nvidia-smi -l 1` to watch GPU usage
6. **Save Often**: Checkpoints are automatic but verify they're being created
7. **Compare Scenarios**: Not all scenarios are equal - some have broken rewards

## Advanced: Resuming Training

To resume from a checkpoint with modified hyperparameters:

```python
# Load existing model
model = PPO.load("checkpoints/scenario/best_model.zip", env=env)

# Update hyperparameters (some are fixed, like n_steps)
model.learning_rate = 3e-4
model.n_epochs = 4
model.ent_coef = 0.01

# Continue training
model.learn(
    total_timesteps=5_000_000,
    callback=callbacks,
    reset_num_timesteps=False  # Continue counting from checkpoint
)
```

See `scripts/resume_defend_center_fast.py` for full example.

## Advanced: Reward Shaping

For scenarios with poor default rewards, wrap the environment:

```python
class CustomRewardWrapper(gym.Wrapper):
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Reshape rewards
        shaped_reward = reward * 0.1  # Scale down
        if terminated:
            shaped_reward -= 500  # Heavy death penalty

        return obs, shaped_reward, terminated, truncated, info
```

See `scripts/train_deadly_corridor_harsh.py` for full example (though it still didn't work for that scenario).

## Recommended Training Path

1. **Start**: Train Take Cover to verify setup works
   - Should reach 500+ reward in 1-2M steps
   - Fast training (~2-3 hours for 2M steps)

2. **Next**: Try Defend The Center for more complexity
   - Expect 20-30 reward after 2M steps
   - May plateau - this is normal

3. **Advanced**: Experiment with other scenarios
   - Health Gathering (navigation + survival)
   - Avoid Deadly Corridor (broken rewards)

4. **Expert**: Try curriculum learning or custom reward shaping

## Project Credits

- **VizDoom**: Farama Foundation
- **Stable Baselines 3**: DLR-RM
- **PyTorch**: Meta AI
- **Development**: Trained and tested on NVIDIA RTX 3090

## Next Steps

After successful training:

1. Record videos of your best agents
2. Try curriculum learning (use simpler scenario as pretraining)
3. Experiment with recurrent policies (LSTM) for memory-based tasks
4. Implement custom CNN architectures
5. Try multi-agent scenarios
6. Share your trained models!

## License

This project is for educational and research purposes.

---

**Last Updated:** January 2026
**Tested On:** Windows 10/11, Python 3.10+, RTX 3090
