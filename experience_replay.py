import numpy as np
import random
from collections import namedtuple, deque
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple(
            "Experience",
            field_names=["state", "action", "reward", "next_state", "done"]
            )
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([
            e.state for e in experiences if e is not None
            ])).float().to(device)
        actions = torch.from_numpy(np.vstack([
            e.action for e in experiences if e is not None
            ])).float().to(device)
        rewards = torch.from_numpy(np.vstack([
            e.reward for e in experiences if e is not None
            ])).float().to(device)
        next_states = torch.from_numpy(np.vstack([
            e.next_state for e in experiences if e is not None
            ])).float().to(device)
        dones = torch.from_numpy(np.vstack([
            e.done for e in experiences if e is not None
            ]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class PriorReplayBuffer:
    """Naive Prioritized buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size,
                 seed, alpha, beta):
        """Initialize a Prioritized ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            alpha (float): exponent to determine how much prioritization is
                going to be used
            beta (float): bias correction exponent
            seed (int): random seed
        """
        self.exp_alpha = alpha
        self.exp_beta = beta
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple(
            "Experience",
            field_names=["priority", "state", "action", "reward",
                         "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        max_priority = np.vstack(
            [e.priority for e in self.memory if e is not None]
            ).max() if self.memory else 1.0
        e = self.experience(
            max_priority, state, action,
            reward, next_state, done
            )
        self.memory.append(e)

    def sample(self):
        """Sample considering probabilities weighted by the TD loss"""
        priorities = np.vstack([
            e.priority for e in self.memory if e is not None
            ])

        probs = priorities.reshape(-1) ** self.exp_alpha
        probs /= probs.sum()

        idxs = np.random.choice(len(self.memory), self.batch_size, p=probs)
        samples = [self.memory[idx] for idx in idxs]

        states = torch.from_numpy(np.vstack([
            e.state for e in samples if e is not None
            ])).float().to(device)
        actions = torch.from_numpy(np.vstack([
            e.action for e in samples if e is not None
            ])).long().to(device)
        rewards = torch.from_numpy(np.vstack([
            e.reward for e in samples if e is not None
            ])).float().to(device)
        next_states = torch.from_numpy(np.vstack([
            e.next_state for e in samples if e is not None
            ])).float().to(device)
        dones = torch.from_numpy(np.vstack([
            e.done for e in samples if e is not None
            ]).astype(np.uint8)).float().to(device)

        N = len(self.memory)
        weights = ((1/N) * (1/probs[idxs])) ** (self.exp_beta)
        weights /= weights.max()
        weights = torch.from_numpy(np.vstack(weights)).float().to(device)

        return (states, actions, rewards, next_states, dones, idxs, weights)

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.memory[idx] = self.memory[idx]._replace(priority=prio[0])

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class Storage:
    def __init__(self, size, id_list, keys=None):
        if keys is None:
            keys = []
        keys = keys + id_list
        self.keys = keys
        self.size = size
        self.reset()

    def add(self, data):
        for k, v in data.items():
            if k not in self.keys:
                self.keys.append(k)
                setattr(self, k, [])
            getattr(self, k).append(v)

    def placeholder(self):
        for k in self.keys:
            v = getattr(self, k)
            if len(v) == 0:
                setattr(self, k, [None] * self.size)

    def reset(self):
        for key in self.keys:
            setattr(self, key, [])

    def cat(self, keys):
        data = [getattr(self, k)[:self.size] for k in keys]
        return map(lambda x: torch.cat(x, dim=0), data)
