import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

class QuantileActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, n_quantiles=32):
        super(QuantileActorCritic, self).__init__()
        self.n_quantiles = n_quantiles
        
        self.common = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        
        # Policy (Actor)
        self.actor = nn.Linear(64, action_dim)
        
        # Distributional Value (Critic) - Outputs N quantiles
        self.critic = nn.Linear(64, n_quantiles)

    def forward(self, x):
        x = self.common(x)
        logits = self.actor(x)
        quantiles = self.critic(x)
        return logits, quantiles

class PPOAgent:
    """
    Quantile Regression PPO (QR-PPO) for Distributional RL.
    Learns the full distribution of execution costs for risk management.
    """
    def __init__(self, state_dim, action_dim, n_quantiles=32, lr=1e-4, gamma=0.99, K_epochs=4, eps_clip=0.2):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.n_quantiles = n_quantiles
        
        self.policy = QuantileActorCritic(state_dim, action_dim, n_quantiles)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.policy_old = QuantileActorCritic(state_dim, action_dim, n_quantiles)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.memory = []

    def act(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            logits, _ = self.policy_old(state)
            
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        
        return action.item(), dist.log_prob(action)

    def store(self, state, action, log_prob, reward, done):
        self.memory.append((state, action, log_prob, reward, done))

    def quantile_huber_loss(self, target, current):
        """
        Quantile Huber Loss for distributional reinforcement learning.
        """
        # target: [batch, n_quantiles], current: [batch, n_quantiles]
        # Reshape for pairwise comparison
        target = target.unsqueeze(2) # [batch, n_quantiles, 1]
        current = current.unsqueeze(1) # [batch, 1, n_quantiles]
        
        diff = target - current # [batch, n_quantiles, n_quantiles]
        huber_loss = torch.where(diff.abs() < 1, 0.5 * diff.pow(2), diff.abs() - 0.5)
        
        # Quantile thresholds (tau)
        tau = torch.linspace(0.0, 1.0, self.n_quantiles + 1)[1:] - (0.5 / self.n_quantiles)
        tau = tau.view(1, 1, self.n_quantiles).to(current.device)
        
        # Weighted loss
        loss = (torch.abs(tau - (diff < 0).float()) * huber_loss).mean(1).sum(1)
        return loss.mean()

    def update(self, alpha=0.95):
        if not self.memory:
            return

        states, actions, log_probs_old, rewards, dones = zip(*self.memory)
        
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        log_probs_old = torch.stack(log_probs_old).squeeze().detach()
        
        # Calculate Returns
        returns = []
        discounted_reward = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            returns.insert(0, discounted_reward)
            
        returns = torch.FloatTensor(returns)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-7)

        for _ in range(self.K_epochs):
            logits, quantiles = self.policy(states)
            
            # Policy Advantage Calculation
            # In Distributional RL, we use the mean of the quantiles as the state value
            state_values = quantiles.mean(dim=1)
            
            dist = torch.distributions.Categorical(logits=logits)
            log_probs = dist.log_prob(actions)
            dist_entropy = dist.entropy()
            
            ratios = torch.exp(log_probs - log_probs_old)
            advantages = (returns - state_values.detach()).detach()
            
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            
            # Distributional Value Loss
            # Target quantiles for returns (simplified: each return is a delta at its value)
            target_quantiles = returns.unsqueeze(1).repeat(1, self.n_quantiles)
            value_loss = self.quantile_huber_loss(target_quantiles, quantiles)
            
            # Risk-Sensitive Penalty (CVaR)
            # The bottom alpha percentile of the learned distribution
            var_idx = int((1 - alpha) * self.n_quantiles)
            sorted_quantiles, _ = torch.sort(quantiles, dim=1)
            cvar = sorted_quantiles[:, :var_idx].mean(dim=1).mean()
            
            # Total Loss
            loss = -torch.min(surr1, surr2).mean() + 0.5 * value_loss - 0.01 * dist_entropy.mean()
            
            # Maximize CVaR (minimize negative CVaR)
            loss_total = loss - 0.1 * cvar
            
            if torch.isnan(loss_total):
                continue

            self.optimizer.zero_grad()
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            self.optimizer.step()
            
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.memory = []
