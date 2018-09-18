import random
from collections import namedtuple, deque
import torch
import numpy as np
class ReplayBuffer:

    def __init__(self, num_actions, buffer_size, batch_size, seed, device):
        """

        :param num_actions:
        :param buffer_size:
        :param batch_size:
        :param seed:
        """

        self.num_actions = num_actions
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.device = device
        self.seed = random.seed(seed)

        self.memory = deque(maxlen=buffer_size)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        """

        :param state:
        :param action:
        :param reward:
        :param next_state:
        :param done:
        :return:
        """

        e = self.experience(state, action, reward, next_state, done)

        self.memory.append(e)

    def sample(self):
        """
        Random sample a batch from the memory
        :return:
        """

        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)