import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os
import glob
from collections import deque
from typing import List
from ..components.vehicle import Vehicle, VehicleStatus
from ..components.incident import Incident


class DQN(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        return self.network(x)


class ReplayBuffer:
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (np.array(state), np.array(action), np.array(reward),
                np.array(next_state), np.array(done))

    def __len__(self):
        return len(self.buffer)


class DispatchAgent:
    def __init__(self, state_dim: int, action_dim: int, learning_rate: float = 0.001,
                 shift_mode: bool = False, auto_load: bool = True):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.shift_mode = shift_mode

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.q_network = DQN(state_dim, action_dim).to(self.device)
        self.target_network = DQN(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

        if shift_mode:
            self.memory = ReplayBuffer(capacity=50000)
            self.epsilon = 0.8
            self.epsilon_min = 0.05
            self.epsilon_decay = 0.9995
            self.batch_size = 64
            self.gamma = 0.995
            self.target_update_frequency = 200
        else:
            self.memory = ReplayBuffer(capacity=10000)
            self.epsilon = 1.0
            self.epsilon_min = 0.01
            self.epsilon_decay = 0.995
            self.batch_size = 32
            self.gamma = 0.99
            self.target_update_frequency = 100

        self.update_counter = 0
        self.update_target_network()

        if auto_load:
            self._try_load_best_model()

    def encode_state(self, vehicles: List[Vehicle], incidents: List[Incident],
                    city_graph, current_time: float) -> np.ndarray:
        state_vector = []

        max_vehicles = 10
        max_incidents = 5

        status_mapping = {
            "idle": 0.0,
            "en_route_to_incident": 1.0,
            "at_incident": 2.0,
            "transporting_to_hospital": 3.0,
            "returning_to_station": 4.0,
            "out_of_service": 5.0
        }

        for i in range(max_vehicles):
            if i < len(vehicles):
                vehicle = vehicles[i]
                status_value = status_mapping.get(vehicle.status.value, 0.0)
                state_vector.extend([
                    vehicle.current_location.x / city_graph.width,
                    vehicle.current_location.y / city_graph.height,
                    status_value / 5.0,
                    1.0 if vehicle.assigned_incident else 0.0
                ])
            else:
                state_vector.extend([0.0, 0.0, 0.0, 0.0])

        for i in range(max_incidents):
            if i < len(incidents):
                incident = incidents[i]
                age = current_time - incident.time_created
                state_vector.extend([
                    incident.location.x / city_graph.width,
                    incident.location.y / city_graph.height,
                    age / 100.0,
                    1.0 if incident.assigned_vehicle else 0.0
                ])
            else:
                state_vector.extend([0.0, 0.0, 0.0, 0.0])

        state_vector.append(current_time / 1000.0)
        state_vector.append(len([v for v in vehicles if v.status == VehicleStatus.IDLE]) / max_vehicles)
        state_vector.append(len(incidents) / max_incidents)

        traffic_state = city_graph.get_traffic_state_vector()
        state_vector.extend(traffic_state)

        return np.array(state_vector, dtype=np.float32)

    def get_valid_actions(self, vehicles: List[Vehicle], incidents: List[Incident]) -> List[int]:
        valid_actions = []

        if not vehicles:
            return [0]

        if incidents:
            for i, vehicle in enumerate(vehicles):
                if vehicle.status == VehicleStatus.IDLE:
                    valid_actions.append(i)

        if not valid_actions:
            return [0] if len(vehicles) > 0 else [0]

        return valid_actions

    def get_action(self, state: np.ndarray, vehicles: List[Vehicle],
                  incidents: List[Incident], training: bool = True) -> int:
        valid_actions = self.get_valid_actions(vehicles, incidents)

        if training and random.random() < self.epsilon:
            return random.choice(valid_actions)

        try:
            if len(state) != self.state_dim:
                print(f"Warning: State dimension mismatch. Expected {self.state_dim}, got {len(state)}")
                return valid_actions[0]

            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)

            masked_q_values = q_values.clone()
            mask = torch.full_like(masked_q_values, float('-inf'))

            for action in valid_actions:
                if action < masked_q_values.size(1):
                    mask[0, action] = 0

            masked_q_values += mask

            action = masked_q_values.argmax().item()
            return action if action in valid_actions and action < len(vehicles) else valid_actions[0]
        except Exception as e:
            print(f"Error in get_action: {e}")
            return valid_actions[0] if valid_actions else 0

    def train_step(self, state: np.ndarray, action: int, reward: float,
                  next_state: np.ndarray, done: bool):
        self.memory.push(state, action, reward, next_state, done)

        if len(self.memory) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)

        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))

        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)

        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_counter += 1
        if self.update_counter % self.target_update_frequency == 0:
            self.update_target_network()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def calculate_reward(self, incident: Incident, response_time: float,
                        system_state: dict) -> float:
        reward = -response_time / 60.0
        print(f"    Reward calculation: {response_time:.1f}s → {reward:.4f}")
        return reward

    def save_model(self, filepath: str):
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'update_counter': self.update_counter
        }, filepath)

    def load_model(self, filepath: str):
        try:
            checkpoint = torch.load(filepath, map_location=self.device)
            self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
            self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint['epsilon']
            self.update_counter = checkpoint['update_counter']
            print(f"Loaded model from {filepath} (epsilon: {self.epsilon:.3f})")
            return True
        except Exception as e:
            print(f"Failed to load model from {filepath}: {e}")
            return False

    def _try_load_best_model(self):
        models_dir = "models"
        if not os.path.exists(models_dir):
            return False

        model_pattern = "shift" if self.shift_mode else "episode"

        patterns_to_try = [
            f"{models_dir}/dqn_{model_pattern}_final.pth",
            f"{models_dir}/dqn_final.pth",
            f"{models_dir}/dqn_{model_pattern}_*.pth",
            f"{models_dir}/dqn_*.pth"
        ]

        for pattern in patterns_to_try:
            if "*" in pattern:
                matching_files = glob.glob(pattern)
                if matching_files:
                    latest_file = max(matching_files, key=os.path.getmtime)
                    if self.load_model(latest_file):
                        return True
            else:
                if os.path.exists(pattern):
                    if self.load_model(pattern):
                        return True

        print(f"No pre-trained model found for {'shift' if self.shift_mode else 'episode'} mode")
        return False

    def get_model_info(self):
        models_dir = "models"
        if not os.path.exists(models_dir):
            return "No models directory"

        models = []
        for file in os.listdir(models_dir):
            if file.endswith('.pth'):
                filepath = os.path.join(models_dir, file)
                size = os.path.getsize(filepath)
                mtime = os.path.getmtime(filepath)
                models.append(f"{file} ({size//1024}KB)")

        return f"Available models: {', '.join(models)}" if models else "No trained models found"