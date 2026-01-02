import os
import math
import random
import pickle

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import deque

import config

class ReplayBuffer:
    def __init__(self, capacity=config.REPLAY_BUFFER_SIZE, alpha=0.6, beta=0.4):
        self.capacity = capacity
        self.states = deque(maxlen=capacity)
        self.policies = deque(maxlen=capacity)
        self.values = deque(maxlen=capacity)
        
        # 优先级采样参数
        self.alpha = alpha
        self.beta = beta
        self.priorities = deque(maxlen=capacity)
        self.max_priority = 1.0
        
    def push(self, state, policy, value, priority=None):
        if priority is None:
            priority = self.max_priority
        
        self.states.append(state)
        self.policies.append(policy)
        self.values.append(value)
        self.priorities.append(priority)
        
        # 更新最大优先级
        if priority > self.max_priority:
            self.max_priority = priority
    
    def sample(self, batch_size):
        if len(self.states) < batch_size:
            return None
        
        # 优先级采样
        priorities = np.array(self.priorities, dtype=np.float32)
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.states), batch_size, p=probs, replace=False)
        
        # 计算重要性采样权重
        total = len(self.states)
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        
        states = np.array([self.states[i] for i in indices], dtype=np.float32)
        policies = np.array([self.policies[i] for i in indices], dtype=np.float32)
        values = np.array([self.values[i] for i in indices], dtype=np.float32)
        
        return states, policies, values, indices, weights
    
    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            if 0 <= idx < len(self.priorities):
                self.priorities[idx] = priority + 1e-6  # 防止为零
                if priority > self.max_priority:
                    self.max_priority = priority
    
    def __len__(self):
        return len(self.states)
    
    def save(self, path):
        print(f"[Buffer] Saving {len(self.states)} samples to {path}...")
        
        # 创建安全的保存数据
        save_data = {
            "states": [],
            "policies": [],
            "values": [],
            "priorities": [],
            "max_priority": float(self.max_priority),
        }
        
        # 转换numpy数组为列表
        for i in range(len(self.states)):
            save_data["states"].append(self.states[i].tolist())
            save_data["policies"].append(self.policies[i].tolist())
            save_data["values"].append(float(self.values[i]))
            save_data["priorities"].append(float(self.priorities[i]))
        
        # 使用pickle保存
        with open(path, "wb") as f:
            pickle.dump(save_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        print(f"[Buffer] Saved {len(self.states)} samples.")

    def load(self, path):
        """加载缓冲区 - 修复兼容性问题"""
        if not os.path.exists(path):
            print("[Buffer] No buffer file found, starting empty.")
            return
        
        print(f"[Buffer] Loading from {path}...")
        
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            
            # 转换回numpy数组
            import numpy as np
            self.states = deque([np.array(s, dtype=np.float32) for s in data["states"]], 
                        maxlen=self.capacity)
            self.policies = deque([np.array(p, dtype=np.float32) for p in data["policies"]], 
                        maxlen=self.capacity)
            self.values = deque([float(v) for v in data["values"]], 
                           maxlen=self.capacity)
            self.priorities = deque([float(p) for p in data["priorities"]], 
                               maxlen=self.capacity)
            self.max_priority = float(data.get("max_priority", 1.0))
            
            print(f"[Buffer] Loaded {len(self.states)} samples.")
        except Exception as e:
            print(f"[Buffer] Load failed: {e}")
            print("[Buffer] Starting with empty buffer.")


def apply_symmetry_batch(states, policies):
    """应用棋盘对称性增强数据"""
    B = states.shape[0]
    states = states.reshape(B, 3, config.BOARD_SIZE, config.BOARD_SIZE)
    policies = policies.reshape(B, config.BOARD_SIZE, config.BOARD_SIZE)
    
    # 随机选择对称变换
    k = np.random.randint(0, 4)  # 旋转次数
    flip = np.random.randint(0, 2)  # 是否翻转
    
    states = np.rot90(states, k, axes=(2, 3))
    policies = np.rot90(policies, k, axes=(1, 2))
    
    if flip:
        states = np.flip(states, axis=3)
        policies = np.flip(policies, axis=2)
    
    states = np.ascontiguousarray(states)
    policies = np.ascontiguousarray(policies)
    
    states = states.reshape(B, 3, config.BOARD_SIZE, config.BOARD_SIZE)
    policies = policies.reshape(B, -1)
    
    return states, policies

def get_amp_context(use_amp=True, device_type='cuda'):
    """获取兼容的AMP上下文管理器"""
    if not use_amp or config.DEVICE_TRAIN.type != 'cuda':
        # 创建假上下文管理器
        class DummyAutocast:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
        return DummyAutocast()
    
    # 检测PyTorch版本并选择正确的autocast
    torch_version = torch.__version__
    major_version = int(torch_version.split('.')[0]) if '.' in torch_version else 1
    
    try:
        # 尝试使用新的torch.amp.autocast
        if hasattr(torch, 'amp') and hasattr(torch.amp, 'autocast'):
            if major_version >= 2 and torch_version >= '2.7.0':
                return torch.amp.autocast(device_type=device_type, dtype=torch.float16)
            else:
                return torch.amp.autocast(enabled=True)
    except:
        pass
    
    try:
        # 回退到torch.cuda.amp.autocast
        if hasattr(torch.cuda, 'amp') and hasattr(torch.cuda.amp, 'autocast'):
            return torch.cuda.amp.autocast()
    except:
        pass
    
    # 最后创建假上下文
    class DummyAutocast:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
    return DummyAutocast()

def train_step(model, optimizer, scaler, buffer, use_amp=True):
    """单次训练步骤，添加标签平滑和稳定性措施"""
    data = buffer.sample(config.BATCH_SIZE)
    if not data:
        return None, None, None
    
    state, policy_t, value_t, indices, weights = data
    
    # 数据增强：应用对称性
    state, policy_t = apply_symmetry_batch(state, policy_t)
    
    # 转换为张量
    state = torch.tensor(state, dtype=torch.float32, device=config.DEVICE_TRAIN)
    policy_t = torch.tensor(policy_t, dtype=torch.float32, device=config.DEVICE_TRAIN)
    value_t = torch.tensor(value_t, dtype=torch.float32, device=config.DEVICE_TRAIN).unsqueeze(1)
    weights = torch.tensor(weights, dtype=torch.float32, device=config.DEVICE_TRAIN).unsqueeze(1)
    
    model.train()
    optimizer.zero_grad()
    
    # 使用兼容的AMP上下文管理器
    amp_context = get_amp_context(use_amp and config.DEVICE_TRAIN.type == 'cuda')
    
    with amp_context:
        policy, value = model(state)
        
        # 标签平滑：防止过拟合到噪声标签
        if hasattr(config, 'LABEL_SMOOTHING') and config.LABEL_SMOOTHING > 0:
            uniform = torch.ones_like(policy_t) / policy_t.size(1)
            policy_t_smoothed = (1 - config.LABEL_SMOOTHING) * policy_t + config.LABEL_SMOOTHING * uniform
        else:
            policy_t_smoothed = policy_t
        
        # 计算策略损失（带重要性采样权重）
        # 使用KL散度代替交叉熵，更稳定
        policy_loss = F.kl_div(torch.log(policy + 1e-8), policy_t_smoothed, reduction='none').sum(dim=1)
        policy_loss = (policy_loss * weights.squeeze()).mean()
        
        # 计算价值损失 - 使用Huber损失，对异常值更鲁棒
        value_loss = F.smooth_l1_loss(value, value_t, reduction='none')
        value_loss = (value_loss * weights).mean()
        
        # 总损失
        total_loss = policy_loss + config.VALUE_LOSS_WEIGHT * value_loss
    
    # 反向传播
    if use_amp and config.DEVICE_TRAIN.type == 'cuda' and hasattr(scaler, 'scale'):
        # 使用scaler进行混合精度训练
        scaler.scale(total_loss).backward()
        
        # 梯度裁剪
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
        
        scaler.step(optimizer)
        scaler.update()
    else:
        # 常规训练
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
        optimizer.step()
    
    # 更新优先级（使用TD误差）
    with torch.no_grad():
        td_errors = torch.abs(value - value_t).cpu().numpy().flatten()
        buffer.update_priorities(indices, td_errors)
    
    return total_loss.item(), policy_loss.item(), value_loss.item()


def evaluate_position(model, board_state, num_simulations=100):
    """评估特定位置"""
    model.eval()
    with torch.no_grad():
        state_tensor = torch.from_numpy(board_state).unsqueeze(0).to(config.DEVICE_TRAIN)
        policy, value = model(state_tensor)
    
    return policy.cpu().numpy()[0], value.item()