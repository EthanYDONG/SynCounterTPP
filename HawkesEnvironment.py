import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argparse
import os
import sys
from hawkes.hawkes import hawkes, hawkes_calculate, sampleHawkes, plotHawkes, iterative_sampling, extract_samples, sample_counterfactual_superposition, check_monotonicity_hawkes
from sampling_utils import thinning_T, return_samples
from counterfactual_tpp import sample_counterfactual, check_monotonicity, distance, covariance
from multiprocessing import cpu_count, Pool
from tqdm import tqdm
import utils


# The simulated HawkesEnvironment represents a patient's illness, controlled by a latent Hawkes process. When an event occurs, it signifies that the patient's illness has worsened once, resulting in a reduction of 1 in the patient's health score, which corresponds to the reward in the reinforcement learning context. At this point, the doctor decides whether to give medicine to the patient. If medicine is used, the alpha parameter(self-exciting paramter) of the Hawkes process governing the patient's illness is reduced by 1/2. However, due to the side effects of the medicine, the patient's health score also decreases by 0.5. The total duration of the simulation is T = 30, we collect the score of patient's health at that time as the return of this episode.

class HawkesEnvironment:
    def __init__(self, mu0, alpha, w, T, max_actions=2):
        self.mu0 = mu0
        self.alpha = alpha
        self.w = w
        self.T = T
        self.max_actions = max_actions
        self.reset()
        self.lambda_max = 1.5

    def reset(self):
        self.time = 0
        self.actions_taken = 0
        self.history = []


    def step(self, action):     
    # Question: Do we have other ways to add constrains in RL algorithms other than using penalty term like ICML'23 Workshop paper
    #     # if self.actions_taken >= self.max_actions:
    #     #     raise ValueError("Treatment can no longer be conducted")
    #      self.actions_taken += 1

        # sample illness occurrence times from Hawkes process
        initial_sample, indicators = thinning_T(0, lambda t: self.mu0, self.lambda_max, self.T)
        events = {initial_sample[i]: indicators[i] for i in range(len(initial_sample))}
        all_events = {}
        all_events[self.mu0] = events
        iterative_sampling(all_events, events, self.mu0, self.alpha, self.w, self.lambda_max, T)
        sampled_events = list(all_events.keys())[1:]
        sampled_events.sort()
        sampled_events = np.array(sampled_events)
        counters = []

    
        if sampled_events[-1] >= self.T:
            raise ValueError("Simulation duration exceeded")
       
        for event_time in events:
            self.health_score -= 1
            # if use drug，the alpha parameter in Hawkes is reduced by 1/2 and health score also reduce 0.5
            # TO DO: action should be a function of the state (which in this case is past evetns)
            if action:  
                new_alpha = self.alpha / 2
                self.health_score -= 0.5
                self.medicine_administered.append(event_time)

        # TO DO: here we might resort to Ogata's modified thinning algorithm for dynamically counterfactual sampling, otherwise previous events + action could not correctly affect future couterfactual events.

              
        # real_counterfactuals = sample_counterfactual_superposition(self.mu0, alpha, new_mu0, new_alpha, all_events, lambda_max, w, T)
        # counters.append(real_counterfactuals)

        done = (self.time >= self.T) or (self.health_score <= -10)  # if score < -10 --> end

        return self.health_score, done


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 16)
        self.fc2 = nn.Linear(16, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=-1)
        return x

# REINFORCE main algorithm
class REINFORCE:
    def __init__(self, policy_network, optimizer):
        self.policy_network = policy_network
        self.optimizer = optimizer

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        action_probs = self.policy_network(state)
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        return action.item(), log_prob

    def update(self, rewards, log_probs):
        discounted_rewards = self._calculate_discounted_rewards(rewards)
        # TO DO: think a more realistic reward function
        policy_loss = (-log_probs * discounted_rewards).sum()
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

    def _calculate_discounted_rewards(self, rewards, gamma=0.99):
        discounted_rewards = []
        R = 0
        for r in rewards[::-1]:
            R = r + gamma * R
            discounted_rewards.insert(0, R)
        return torch.tensor(discounted_rewards)

# Train policy network
def train_policy_network(env, policy_network, optimizer, num_episodes=1000, max_steps=100):
    reinforce = REINFORCE(policy_network, optimizer)
    for episode in range(num_episodes):
        env.reset()
        rewards = []
        log_probs = []
        for step in range(max_steps):
            state = env.history
            action, log_prob = reinforce.select_action(state)
            next_state, reward, done = env.step(action)
            rewards.append(reward)
            log_probs.append(log_prob)
            if done:
                break
        reinforce.update(rewards, log_probs)
        if episode % 100 == 0:
            print(f"Episode {episode}, Reward: {reward}")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train Reinforcement Learning policy on Hawkes environment")
    parser.add_argument("--mu0", type=float, default=1, help="Initial intensity parameter for Hawkes process")
    parser.add_argument("--alpha", type=float, default=1, help="Alpha parameter for Hawkes process")
    parser.add_argument("--w", type=float, default=1, help="Parameter w for Hawkes process")
    parser.add_argument("--T", type=int, default=10, help="Total duration of the simulation")
    parser.add_argument("--max_actions", type=int, default=2, help="Maximum number of actions allowed")
    return parser.parse_args()

if __name__ == '__main__':

    args = parse_arguments()

    # Create environment and policy network
    env = HawkesEnvironment(args.mu0, args.alpha, args.w, args.T, args.max_actions)
    state_dim = args.max_actions  # Maximum number of actions
    action_dim = 2  # Action space (0: not use medicine or 1:use medicine)
    policy_network = PolicyNetwork(state_dim, action_dim)
    optimizer = optim.Adam(policy_network.parameters(), lr=0.001)

    train_policy_network(env, policy_network, optimizer)