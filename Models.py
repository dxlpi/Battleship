import random
import numpy as np
from tqdm import tqdm
from abc import abstractmethod
from collections import defaultdict, deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class Models:
    def __init__(self):
        self.unvisited_location = [i for i in range(100)]

    def reset(self):
        self.unvisited_location = [i for i in range(100)]

    @abstractmethod
    def run(self):
        pass

class TDModel(Models):
    def __init__(self, env, num_episodes=20000, epsilon=0.1, alpha=0.5, gamma=0.9):
        super().__init__()

        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.num_episodes = num_episodes
        self.env = env
        self.q_table = defaultdict(lambda : [0] * 100)

    def run(self):
        rewards = []

        for _ in tqdm(range(self.num_episodes)):
            self.env.reset()
            self.env.start()
            super().reset()

            reward = 0
            state = str(self.env.current_state())

            while not self.env.end():
                if np.random.rand() <= self.epsilon:
                    next_location = random.choice(self.unvisited_location)
                else:
                    possible_location  = [self.q_table[state][location] for location in self.unvisited_location]
                    next_index = np.argmax(possible_location)
                    next_location = self.unvisited_location[next_index]

                self.unvisited_location.remove(next_location)
                cur_reward = self.env.step(next_location // 10, next_location % 10)
                next_state = str(self.env.current_state())

                self.q_table[state][next_location] += self.alpha * (cur_reward + self.gamma * max(self.q_table[next_state]) - self.q_table[state][next_location])

                reward += cur_reward
                state = next_state

            rewards.append(reward)

        return self.q_table, rewards

class MCModel(Models):
    def __init__(self, env, num_episodes=20000, epsilon=0.1, gamma=0.9):
        super().__init__()

        self.epsilon = epsilon
        self.gamma = gamma
        self.num_episodes = num_episodes
        self.env = env
        self.q_table = defaultdict(lambda: [0] * 100)
        self.returns = defaultdict(lambda: [0] * 100)
        self.visit_counts = defaultdict(lambda: [0] * 100)

    def run(self):
        rewards = []

        for _ in tqdm(range(self.num_episodes)):
            self.env.reset()
            self.env.start()
            super().reset()
            self.returns = defaultdict(lambda: [0] * 100)
            self.visit_counts = defaultdict(lambda: [0] * 100)

            episode = []
            reward = 0
            state = str(self.env.current_state())

            while not self.env.end():
                if np.random.rand() <= self.epsilon:
                    next_location = random.choice(self.unvisited_location)
                else:
                    possible_location = [self.q_table[state][location] for location in self.unvisited_location]
                    next_index = np.argmax(possible_location)
                    next_location = self.unvisited_location[next_index]

                episode.append((state, next_location))
                self.unvisited_location.remove(next_location)
                cur_reward = self.env.step(next_location // 10, next_location % 10)
                next_state = str(self.env.current_state())

                reward += cur_reward
                state = next_state

            G = 0
            for state, location in reversed(episode):
                G = self.gamma * G + reward
                self.returns[state][location] += G
                self.visit_counts[state][location] += 1
                self.q_table[state][location] = self.returns[state][location] / self.visit_counts[state][location]

            rewards.append(reward)

        return self.q_table, rewards


class DQNModel(Models):
    def __init__(self, env, num_episodes=2000, epsilon=0.1, alpha=0.5, gamma=0.9, batch_size=64, buffer_size=10000):
        super().__init__()

        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.num_episodes = num_episodes
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.env = env

        self.q_network = self.build_q_network()
        self.target_network = self.build_q_network()
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.001)

        self.replay_buffer = deque(maxlen=self.buffer_size)

    def build_q_network(self):
        class QNetwork(nn.Module):
            def __init__(self):
                super(QNetwork, self).__init__()
                self.fc1 = nn.Linear(100, 128)
                self.fc2 = nn.Linear(128, 128)
                self.output = nn.Linear(128, 100)

            def forward(self, x):
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                return self.output(x)

        return QNetwork()

    def store_experience(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def update_q_network(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        q_values = self.q_network(states)

        next_q_values = self.target_network(next_states)
        max_next_q_values = next_q_values.max(dim=1)[0]
        target_q_values = rewards + (self.gamma * max_next_q_values * (1 - dones))

        q_values_taken = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        loss = nn.MSELoss()(q_values_taken, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def run(self):
        rewards = []

        for episode in tqdm(range(self.num_episodes)):
            self.env.reset()
            self.env.start()
            super().reset()

            reward = 0
            state = np.array(self.env.current_state()).flatten()
            done = False

            while not done:
                if np.random.rand() <= self.epsilon:
                    next_location = random.choice(self.unvisited_location)
                else:
                    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                    q_values = self.q_network(state_tensor)
                    next_location = torch.argmax(q_values, dim=1).item()

                    if next_location not in self.unvisited_location:
                        next_location = self.unvisited_location[0]

                self.unvisited_location.remove(next_location)
                next_state = np.array(self.env.current_state()).flatten()
                cur_reward = self.env.step(next_location // 10, next_location % 10)
                done = self.env.end()

                self.store_experience(state, next_location, cur_reward, next_state, done)
                self.update_q_network()

                state = next_state
                reward += cur_reward

            if episode % 100 == 0:
                self.target_network.load_state_dict(self.q_network.state_dict())

            rewards.append(reward)

        return self.q_network, rewards
