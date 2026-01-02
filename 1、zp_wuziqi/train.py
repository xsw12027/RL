import math
import time
import random
import signal
import sys
import os

import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.multiprocessing as mp

def get_amp_modules():
    """获取AMP模块，处理PyTorch版本兼容性"""
    # 检测PyTorch版本
    torch_version = torch.__version__
    major_version = int(torch_version.split('.')[0]) if '.' in torch_version else 1
    
    # 如果PyTorch 2.7+，使用新的torch.amp API
    if hasattr(torch, 'amp') and hasattr(torch.amp, 'autocast') and hasattr(torch.amp, 'GradScaler'):
        try:
            from torch.amp import autocast, GradScaler
            # 创建适合新旧版本的包装器
            class CompatibleAutocast(autocast):
                def __init__(self, enabled=True, device_type='cuda', dtype=None):
                    # 根据PyTorch版本决定参数
                    if major_version >= 2 and torch_version >= '2.7.0':
                        super().__init__(device_type=device_type, dtype=dtype or torch.float16, enabled=enabled)
                    else:
                        super().__init__(enabled=enabled)
                
                def __enter__(self):
                    return super().__enter__()
                
                def __exit__(self, exc_type, exc_val, exc_tb):
                    super().__exit__(exc_type, exc_val, exc_tb)
            
            class CompatibleGradScaler(GradScaler):
                def __init__(self, enabled=True):
                    if major_version >= 2 and torch_version >= '2.7.0':
                        super().__init__(device='cuda', enabled=enabled)
                    else:
                        super().__init__(enabled=enabled)
            
            return CompatibleAutocast, CompatibleGradScaler, True
            
        except Exception as e:
            print(f"[System] New AMP API failed: {e}, falling back...")
    
    # 回退到torch.cuda.amp
    if torch.cuda.is_available():
        try:
            from torch.cuda.amp import autocast, GradScaler
            return autocast, GradScaler, True
        except ImportError as e:
            print(f"[System] CUDA AMP import failed: {e}")
    
    # 最后回退到CPU假实现
    print("[System] Using CPU-compatible AMP implementation")
    
    class CpuAutocast:
        def __init__(self, enabled=True, device_type='cuda'):
            self.enabled = enabled
        
        def __enter__(self):
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            pass
    
    class CpuGradScaler:
        def __init__(self, enabled=True):
            self.enabled = enabled
            self._scale = torch.tensor(1.0)
        
        def scale(self, loss):
            return loss
        
        def step(self, optimizer):
            optimizer.step()
        
        def update(self):
            pass
        
        def unscale_(self, optimizer):
            pass
        
        def state_dict(self):
            return {'_scale': self._scale}
        
        def load_state_dict(self, state_dict):
            if '_scale' in state_dict:
                self._scale = state_dict['_scale']
    
    return CpuAutocast, CpuGradScaler, False

# 导入AMP模块
autocast, GradScaler, amp_available = get_amp_modules()

# 导入配置信息
import config
from environment import *
from agent import *
from tools import *

def save_checkpoint(iteration, model, optimizer, scheduler, scaler, buffer, best_loss):
    """保存检查点 - 使用安全格式"""
    print(f"\n[System] Saving checkpoint for iteration {iteration}...")
    
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    
    # 确保所有数据都是Python原生类型或张量
    safe_state = {
        "iteration": iteration,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "best_loss": float(best_loss),  # 转换为Python float
    }
    
    # 保存检查点 - 使用新版序列化
    torch.save(safe_state, config.CHECKPOINT_FILE, 
               _use_new_zipfile_serialization=True)
    
    # 保存缓冲区
    buffer.save(config.BUFFER_FILE)
    
    print("[System] Save Complete.")


def load_checkpoint(model, optimizer, scheduler, scaler, buffer):
    start_iter = 1
    best_loss = float('inf')
    
    # 首先尝试找到最新的模型文件
    model_files = []
    for f in os.listdir(config.MODEL_DIR):
        if f.startswith('model_iter') and f.endswith('.pth'):
            try:
                iter_num = int(f.split('iter')[1].split('.')[0])
                model_files.append((iter_num, f))
            except:
                continue
    
    # 也检查检查点文件中的迭代次数
    checkpoint_iter = 0
    if os.path.exists(config.CHECKPOINT_FILE):
        try:
            checkpoint = torch.load(config.CHECKPOINT_FILE, 
                                   map_location=config.DEVICE_TRAIN,
                                   weights_only=False)
            checkpoint_iter = checkpoint.get("iteration", 0)
            print(f"[System] Checkpoint iteration: {checkpoint_iter}")
        except:
            pass
    
    # 选择最新的迭代
    max_model_iter = max([iter_num for iter_num, _ in model_files]) if model_files else 0
    latest_iter = max(max_model_iter, checkpoint_iter)
    
    if latest_iter > 0:
        print(f"[System] Found latest iteration: {latest_iter}")
        
        # 如果有检查点且检查点是最新的，优先加载检查点（因为它包含优化器状态）
        if checkpoint_iter >= max_model_iter and os.path.exists(config.CHECKPOINT_FILE):
            print(f"[System] Loading from checkpoint (iteration {checkpoint_iter})")
            
            try:
                # 尝试使用weights_only=False
                checkpoint = torch.load(config.CHECKPOINT_FILE, 
                                       map_location=config.DEVICE_TRAIN,
                                       weights_only=False)
                
                # 加载模型状态
                model.load_state_dict(checkpoint["state_dict"])
                
                # 加载其他状态（如果存在）
                if "optimizer" in checkpoint:
                    optimizer.load_state_dict(checkpoint["optimizer"])
                if "scheduler" in checkpoint:
                    scheduler.load_state_dict(checkpoint["scheduler"])
                if "scaler" in checkpoint:
                    scaler.load_state_dict(checkpoint["scaler"])
                
                best_loss = checkpoint.get("best_loss", float('inf'))
                start_iter = checkpoint.get("iteration", 1) + 1
                
                print(f"[System] Checkpoint loaded successfully")
                
            except Exception as e:
                print(f"[System] Checkpoint loading failed: {e}")
                print("[System] Will try to load model file instead...")
                latest_iter = max_model_iter
        
        # 如果检查点加载失败或模型文件更新，则加载模型文件
        if latest_iter == max_model_iter and model_files:
            # 按迭代次数排序
            model_files.sort(key=lambda x: x[0])
            latest_iter, latest_model = model_files[-1]
            
            print(f"[System] Loading from model file: {latest_model} (iteration {latest_iter})")
            
            try:
                # 加载模型权重
                model_path = os.path.join(config.MODEL_DIR, latest_model)
                model.load_state_dict(torch.load(model_path, map_location=config.DEVICE_TRAIN, weights_only=False))
                
                # 设置开始迭代
                start_iter = latest_iter + 1
                print(f"[System] Model loaded successfully, starting from iteration {start_iter}")
                
                # 注意：这里没有加载优化器状态，所以优化器会重新开始
                # 如果你有保存优化器状态的单独文件，可以在这里加载
                
            except Exception as e:
                print(f"[System] Model loading failed: {e}")
                start_iter = 1
    
    # 加载缓冲区
    buffer.load(config.BUFFER_FILE)
    
    print(f"[System] Starting from iteration: {start_iter}")
    print(f"[System] Best loss: {best_loss}")
    
    return start_iter, best_loss

def evaluate_model(model, num_games=20):
    """评估模型性能"""
    model.eval()
    wins, losses, draws = 0, 0, 0
    
    print("  [Eval] Starting evaluation...")
    
    for game_idx in range(num_games):
        board = Board()
        mcts = MCTS(model, simulations=100)
        
        while True:
            # 当前玩家使用模型
            if board.current_player == 1:
                mcts.clear()
                for _ in range(100):
                    mcts.search(board.copy())
                
                pi = mcts.get_policy(board, temperature=0.1)
                action = np.argmax(pi)
            else:
                # 对手使用随机策略
                legal_moves = board.get_legal_moves()
                action = np.random.choice(legal_moves)
            
            x, y = divmod(action, config.BOARD_SIZE)
            board.move(x, y)
            
            winner = board.check_win()
            if winner != 0:
                if winner == 1:
                    wins += 1
                elif winner == 2:
                    losses += 1
                else:
                    draws += 1
                break
    
    win_rate = wins / num_games
    print(f"  [Eval] Result: {wins}W, {losses}L, {draws}D, Win Rate: {win_rate:.2%}")
    return win_rate


def print_training_diagnostics(model, buffer):
    """打印训练诊断信息"""
    if len(buffer) == 0:
        return
    
    # 抽样一些数据检查价值分布
    sample_size = min(100, len(buffer))
    indices = np.random.choice(len(buffer), sample_size, replace=False)
    
    sample_values = []
    for idx in indices:
        sample_values.append(buffer.values[idx])
    
    values_np = np.array(sample_values)
    print(f"  [Diagnostic] Value stats: min={values_np.min():.3f}, "
          f"max={values_np.max():.3f}, mean={values_np.mean():.3f}, "
          f"std={values_np.std():.3f}")
    
    # 检查策略熵
    sample_policies = []
    for idx in indices:
        sample_policies.append(buffer.policies[idx])
    
    policy_entropies = []
    for policy in sample_policies:
        entropy = -np.sum(policy * np.log(policy + 1e-8))
        policy_entropies.append(entropy)
    
    print(f"  [Diagnostic] Policy entropy: min={min(policy_entropies):.3f}, "
          f"max={max(policy_entropies):.3f}, mean={np.mean(policy_entropies):.3f}")

def main_train():
    """主训练函数"""
    mp.set_start_method("spawn", force=True)
    
    # 创建模型和优化器
    model = AplhaZeroNet(num_residual_blocks=config.RES_BLOCK_NUM, channels=config.CHANNELS).to(config.DEVICE_TRAIN)
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config.LEARNING_RATE, 
        weight_decay=config.WEIGHT_DECAY,
        betas=(0.9, 0.999)
    )
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, 
        milestones=config.LR_MILESTONES, 
        gamma=config.LR_SCHEDULER_GAMMA
    )
    
    # 混合精度训练的scaler
    if torch.cuda.is_available() and config.USE_AMP:
        # 使用兼容的GradScaler
        scaler = GradScaler(enabled=config.USE_AMP)
        print(f"[System] AMP enabled with {'new' if amp_available else 'old'} API")
    else:
        # 创建兼容的假实现
        class DummyScaler:
            def __init__(self):
                self._scale = torch.tensor(1.0)
            
            def scale(self, loss):
                return loss
            
            def step(self, optimizer):
                optimizer.step()
            
            def update(self):
                pass
            
            def unscale_(self, optimizer):
                pass
            
            def state_dict(self):
                return {'_scale': self._scale}
            
            def load_state_dict(self, state_dict):
                if '_scale' in state_dict:
                    self._scale = state_dict['_scale']
        
        scaler = DummyScaler()
        print("[System] AMP disabled, using dummy scaler")
    
    # 缓冲区
    buffer = ReplayBuffer()
    
    # 加载检查点
    start_iter, best_loss = load_checkpoint(model, optimizer, scheduler, scaler, buffer)
    
    
    print(f"\n[System] Starting Training from Iteration {start_iter}...")
    print(f"[System] Device: {config.DEVICE_TRAIN}")
    print(f"[System] Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"[System] Buffer Size: {len(buffer)}")
    print(f"[System] Total Iterations: {config.MAX_ITERS}")
    print(f"[System] Estimated Remaining Time: ~{((config.MAX_ITERS - start_iter + 1) * 15 / 60):.1f} hours")
    
    # 训练循环
    try:
        for iteration in range(start_iter, config.MAX_ITERS + 1):
            # 动态调整模拟次数 - 随着训练进展逐渐增加
            progress = (iteration - 1) / config.MAX_ITERS
            current_sim = int(config.BASE_SIMULATION + progress * (config.MAX_SIMULATION - config.BASE_SIMULATION))
            
            print(f"\n{'='*60}")
            print(f"Iteration {iteration:03d}/{config.MAX_ITERS} | Progress: {progress:.1%}")
            print(f"Sims: {current_sim:03d} | Buffer: {len(buffer):06d} | LR: {optimizer.param_groups[0]['lr']:.2e}")
            print(f"{'='*60}")
            
            iter_start = time.time()
            
            #自对弈阶段
            model.eval()
            shared_model = {k: v.cpu() for k, v in model.state_dict().items()}
            
            print("  [Self-Play] Generating games...")
            self_play_start = time.time()
            
            # 根据缓冲区大小调整工作进程数
            effective_workers = min(config.NUM_WORKERS, config.SELF_PLAY_GAMES)
            
            with mp.Pool(effective_workers, initializer=init_worker, initargs=(shared_model,)) as pool:
                results = list(tqdm(
                    pool.imap_unordered(self_play_worker, [current_sim] * config.SELF_PLAY_GAMES),
                    total=config.SELF_PLAY_GAMES,
                    desc="  Self-Play",
                    bar_format="{l_bar}{bar:30}{r_bar}"
                ))
            
            # 收集数据
            new_samples = 0
            for result in results:
                for state, policy, value in result:
                    buffer.push(state, policy, value)
                    new_samples += 1
            
            self_play_time = time.time() - self_play_start
            print(f"  [Self-Play] Added {new_samples} samples in {self_play_time:.1f}s")
            
            #训练阶段
            if len(buffer) >= config.MIN_BUFFER_TO_TRAIN:
                print("  [Training] Updating model...")
                train_start = time.time()
                
                model.train()
                total_losses, policy_losses, value_losses = [], [], []
                
                # 训练多个epoch
                for epoch in range(config.EPOCHS_PER_UPDATE):
                    loss, policy_loss, value_loss = train_step(
                        model, optimizer, scaler, buffer, use_amp=config.USE_AMP
                    )
                    
                    if loss is not None:
                        total_losses.append(loss)
                        policy_losses.append(policy_loss)
                        value_losses.append(value_loss)
                
                train_time = time.time() - train_start
                
                # 更新学习率
                scheduler.step()
                
                # 打印训练统计
                if total_losses:
                    avg_total = np.mean(total_losses)
                    avg_policy = np.mean(policy_losses)
                    avg_value = np.mean(value_losses)
                    
                    print(f"  [Training] Loss: {avg_total:.4f} (P: {avg_policy:.4f}, V: {avg_value:.4f})")
                    print(f"  [Training] Time: {train_time:.1f}s")
                    
                    # 保存最佳模型
                    if avg_total < best_loss:
                        best_loss = avg_total
                        best_model_path = os.path.join(config.MODEL_DIR, f"best_model_iter{iteration}.pth")
                        torch.save(model.state_dict(), best_model_path)
                        print(f"  [Training] New best model saved to {best_model_path}")
            
            # 诊断和评估
            iter_time = time.time() - iter_start
            
            # 每10次迭代进行诊断
            if iteration % 10 == 0 and len(buffer) >= config.MIN_BUFFER_TO_TRAIN:
                print_training_diagnostics(model, buffer)
            
            # 每20次迭代评估一次
            if iteration % 20 == 0 and len(buffer) >= config.MIN_BUFFER_TO_TRAIN:
                win_rate = evaluate_model(model, num_games=10)
                
                # 如果胜率超过55%，保存一个特别版本
                if win_rate > 0.55:
                    strong_model_path = os.path.join(config.MODEL_DIR, f"strong_model_iter{iteration}.pth")
                    torch.save(model.state_dict(), strong_model_path)
                    print(f"  [Eval] Strong model saved (win rate: {win_rate:.2%})")
            
            # 保存检查点
            # 每25次迭代保存检查点
            if iteration % 25 == 0:
                save_checkpoint(iteration, model, optimizer, scheduler, scaler, buffer, best_loss)
                # 保存当前模型
                current_model_path = os.path.join(config.MODEL_DIR, f"model_iter{iteration}.pth")
                torch.save(model.state_dict(), current_model_path)
            
            # 每50次迭代保存一个轻量版本
            if iteration % 50 == 0:
                light_model_path = os.path.join(config.MODEL_DIR, f"light_model_iter{iteration}.pth")
                torch.save(model.state_dict(), light_model_path)
            
            # 计算剩余时间
            elapsed_time = time.time() - iter_start
            avg_iter_time = elapsed_time if iteration == start_iter else (elapsed_time + (iteration - start_iter - 1) * avg_iter_time) / (iteration - start_iter + 1)
            remaining_iters = config.MAX_ITERS - iteration
            remaining_time = remaining_iters * avg_iter_time
            
            print(f"  [Iteration] Completed in {elapsed_time:.1f}s | Avg: {avg_iter_time:.1f}s")
            print(f"  [Remaining] {remaining_iters} iterations, ~{remaining_time/3600:.1f} hours")
            
            # 如果当前迭代时间太长，给出警告
            if elapsed_time > 1200:  # 超过20分钟
                print(f"  [Warning] Iteration took too long ({elapsed_time:.1f}s > 20min).")
                print("  Consider reducing SELF_PLAY_GAMES or BASE_SIMULATION.")
    
    except KeyboardInterrupt:
        print("\n\n[Warning] Training Interrupted by User (Ctrl+C)!")
        print("Saving current state before exiting...")
        save_checkpoint(iteration, model, optimizer, scheduler, scaler, buffer, best_loss)
        print("Safe to exit now.")
        sys.exit(0)
    
    print(f"\n{'='*60}")
    print("[System] Training Completed!")
    print(f"{'='*60}")
    
    # 保存最终模型
    final_path = os.path.join(config.MODEL_DIR, "final_model.pth")
    torch.save(model.state_dict(), final_path)
    print(f"[System] Final model saved to {final_path}")
    
    # 总结训练结果
    print(f"\n[Summary] Total Iterations: {config.MAX_ITERS}")
    print(f"[Summary] Best Loss: {best_loss:.4f}")
    print(f"[Summary] Models saved in: {config.MODEL_DIR}")