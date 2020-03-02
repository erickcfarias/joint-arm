import progressbar as pb
import numpy as np
import random
import copy
from collections import namedtuple, deque

from model import *
from experience_replay import *

import torch
import torch.nn.functional as F
import torch.optim as optim

from random_process import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class AgentZero:
    def __init__(self):
        pass


class DDPG_Agent(AgentZero):
    """Interacts with and learns from the environment."""

    def __init__(self,
                 state_size,
                 action_size,
                 seed,
                 buffer_size,
                 batch_size,
                 gamma,
                 tau,
                 lr,
                 w_decay
                 ):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        super(AgentZero, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.lr = lr
        self.w_decay = w_decay
        self.noise = OrnsteinUhlenbeckProcess(action_size, seed)

        # Network: Actor
        self.actor_local = DDPG_Actor(state_size, action_size, seed).\
            to(device)
        self.actor_target = DDPG_Actor(state_size, action_size, seed).\
            to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(),
                                          lr=self.lr)

        # Network: Critic
        self.critic_local = DDPG_Critic(state_size, action_size, seed).\
            to(device)
        self.critic_target = DDPG_Critic(state_size, action_size, seed).\
            to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(),
                                           lr=self.lr)

        self.memory = ReplayBuffer(action_size, self.buffer_size,
                                   self.batch_size, seed)

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn if memory size > batch size
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences)

    def act(self, state, add_noise=False):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
        """
        state = torch.from_numpy(state).float().to(device)

        self.actor_local.eval()  # set network to trainable = False
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        if add_noise:
            action += self.noise.sample()
        self.actor_local.train()  # set network to trainable = True

        return action

    def reset(self):
        self.noise.reset_states()

    def learn(self, experiences):
        """Update policy and value parameters using given batch of experience
            tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done)
                tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ------------------ #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)

        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))

        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)

        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---------------------------- update actor ----------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()

        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ------------- #
        self.soft_update(self.critic_local, self.critic_target, self.tau)
        self.soft_update(self.actor_local, self.actor_target, self.tau)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(
            target_model.parameters(), local_model.parameters()
        ):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)
                                    * target_param.data)

    def train(self, env, brain_name, n_episodes):
        """Deep Q-Learning.

        Params
        ======
            n_episodes (int): maximum number of training episodes
        """
        # list containing scores from each episode
        scores = []
        scores_window = deque(maxlen=100)  # last 100 scores
        for i_episode in range(1, n_episodes+1):
            env_info = env.reset(train_mode=True)[brain_name]
            self.reset()
            state = env_info.vector_observations[0]
            score = 0
            while True:
                action = self.act(state)
                # send the action to the environment
                env_info = env.step(action)[brain_name]
                next_state = env_info.vector_observations[0]   # get the next state
                reward = env_info.rewards[0]                   # get the reward
                done = env_info.local_done[0]
                self.step(state, action, reward, next_state, done)
                state = next_state
                score += reward
                if done:
                    break

            scores_window.append(score)       # save most recent score
            scores.append(score)              # save most recent score

            print('\rEpisode {}\tAverage Score: {:.2f}'.format(
                i_episode, np.mean(scores_window)), end="")
            if i_episode % 100 == 0:
                print('\rEpisode {}\tAverage Score: {:.2f}'.format(
                    i_episode, np.mean(scores_window)))

            if np.mean(scores_window) >= 30.0:
                print('\nEnvironment solved in {:d} \
                    episodes!\tAverage Score: {:.2f}'.format(
                    i_episode-100, np.mean(scores_window)))
                torch.save(self.actor_local.state_dict(),
                           'ddpg_checkpoint.pth')
                break

        return scores


class PPO_Agent(AgentZero):
    """Interacts with and learns from the environment."""

    def __init__(self,
                 state_size,
                 action_size,
                 seed,
                 gamma,
                 lr,
                 w_decay
                 ):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        super(PPO_Agent, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.gamma = gamma
        self.lr = lr
        self.w_decay = w_decay

        # Network: Actor
        self.ppo = ModelPPO(state_size, action_size, 128).\
            to(device)
        self.optimizer = optim.Adam(self.ppo.parameters(), lr=self.lr,
                                    weight_decay=self.w_decay)

    def step(self):
        # training loop max iterations
        episode = 500

        envs = parallelEnv('PongDeterministic-v4', n=8, seed=1234)

        discount_rate = .99
        epsilon = 0.1
        beta = .01
        tmax = 320
        SGD_epoch = 4

        # keep track of progress
        mean_rewards = []

        for e in range(episode):

            # collect trajectories
            old_probs, states, actions, rewards = \
                pong_utils.collect_trajectories(envs, policy, tmax=tmax)

            total_rewards = np.sum(rewards, axis=0)

            # gradient ascent step
            for _ in range(SGD_epoch):

                # uncomment to utilize your own clipped function!
                # L = -clipped_surrogate(policy, old_probs, states, actions, rewards, epsilon=epsilon, beta=beta)

                L = -pong_utils.clipped_surrogate(policy, old_probs, states, actions, rewards,
                                                epsilon=epsilon, beta=beta)
                optimizer.zero_grad()
                L.backward()
                optimizer.step()
                del L

            # the clipping parameter reduces as time goes on
            epsilon *= .999

            # the regulation term also reduces
            # this reduces exploration in later runs
            beta *= .995

            # get the average reward of the parallel environments
            mean_rewards.append(np.mean(total_rewards))

            # display some progress every 20 iterations
            if (e+1) % 20 == 0:
                print("Episode: {0:d}, score: {1:f}".format(
                    e+1, np.mean(total_rewards)))
                print(total_rewards)

    def act(self, state):
        pass

    def clipped_surrogate(
        self,
        policy,
        old_probs,
        states,
        actions,
        rewards,
        epsilon=0.1,
        beta=0.01
        ):

        discount = self.gamma**np.arange(len(rewards))
        rewards = np.asarray(rewards)*self.gamma[:, np.newaxis]

        # convert rewards to future rewards and normalize it
        rewards_future = rewards[::-1].cumsum(axis=0)[::-1]

        mean = np.mean(rewards_future, axis=1)
        std = np.std(rewards_future, axis=1) + 1.0e-10

        rewards_normalized = (rewards_future - mean[:, np.newaxis])/std[:, np.newaxis]

        # convert everything into pytorch tensors and move to gpu if available
        actions = torch.tensor(actions, dtype=torch.int8, device=device)
        old_probs = torch.tensor(old_probs, dtype=torch.float, device=device)
        rewards = torch.tensor(rewards_normalized, dtype=torch.float, device=device)

        # convert states to policy (or probability)
        state = torch.from_numpy(state).float().to(device)
        new_probs = self.ppo(states)

        # ratio for clipping
        ratio = new_probs/old_probs

        # clipped function
        clip = torch.clamp(ratio, 1-epsilon, 1+epsilon)
        clipped_surrogate = torch.min(ratio*rewards, clip*rewards)

        # include a regularization term
        # this steers new_policy towards 0.5
        # add in 1.e-10 to avoid log(0) which gives nan
        entropy = -(new_probs*torch.log(old_probs+1.e-10)+ \
            (1.0-new_probs)*torch.log(1.0-old_probs+1.e-10))


        # this returns an average of all the entries of the tensor
        # effective computing L_sur^clip / T
        # averaged over time-step and number of trajectories
        # this is desirable because we have normalized our rewards
        return torch.mean(clipped_surrogate + beta*entropy)


    def collect_trajectories(envs, policy, tmax=200, nrand=5):

        # number of parallel instances
        n = len(envs.ps)

        #initialize returning lists and start the game!
        state_list = []
        reward_list = []
        prob_list = []
        action_list = []

        envs.reset()

        # start all parallel agents
        envs.step([1]*n)

        # perform nrand random steps
        for _ in range(nrand):
            fr1, re1, _, _ = envs.step(np.random.choice([RIGHT, LEFT], n))
            fr2, re2, _, _ = envs.step([0]*n)

        for t in range(tmax):

            # prepare the input
            # preprocess_batch properly converts two frames into
            # shape (n, 2, 80, 80), the proper input for the policy
            # this is required when building CNN with pytorch
            batch_input = preprocess_batch([fr1, fr2])

            # probs will only be used as the pi_old
            # no gradient propagation is needed
            # so we move it to the cpu
            probs = policy(batch_input).squeeze().cpu().detach().numpy()

            action = np.where(np.random.rand(n) < probs, RIGHT, LEFT)
            probs = np.where(action == RIGHT, probs, 1.0-probs)

            # advance the game (0=no action)
            # we take one action and skip game forward
            fr1, re1, is_done, _ = envs.step(action)
            fr2, re2, is_done, _ = envs.step([0]*n)

            reward = re1 + re2

            # store the result
            state_list.append(batch_input)
            reward_list.append(reward)
            prob_list.append(probs)
            action_list.append(action)

            # stop if any of the trajectories is done
            # we want all the lists to be retangular
            if is_done.any():
                break

        # return pi_theta, states, actions, rewards, probability
        return prob_list, state_list, \
            action_list, reward_list


class AgentA2C(AgentZero):

    def __init__(self, env, brain_name, state_size, act_size, gamma,
                 actor_lr, critic_lr, entropy_beta):

        super(AgentA2C, self).__init__()
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.env = env
        self.brain_name = brain_name
        self.state_size = state_size
        self.act_size = act_size
        self.gamma = gamma
        self.entropy_beta = entropy_beta

        self.actor = ActorA2C(self.state_size, self.act_size).\
            to(self.device)
        self.actor_optimizer = optim.RMSprop(
            self.actor.parameters(), lr=actor_lr)

        self.critic = CriticA2C(self.state_size).\
            to(self.device)
        self.critic_optimizer = optim.RMSprop(
            self.critic.parameters(), lr=critic_lr)

    def act(self, state):
        self.actor.eval()
        with torch.no_grad():
            mu, var = self.actor(state)
        dist = torch.distributions.Normal(
            loc=mu,
            scale=torch.sqrt(var)
        )

        action = dist.sample()
        action = torch.clamp(action, -1, 1)
        action = action.cpu().data.numpy()
        self.actor.train()

        return action

    def optimize_models(self, storage):
        # Optimize models
        self.critic_optimizer.zero_grad()
        self.actor_optimizer.zero_grad()

        # Critic Loss
        V, R, A = storage.cat(['action_values', 'returns', 'advantages'])
        value_loss = (R - V).pow(2).mean()

        # Actor Loss
        L_p = torch.tensor(storage.log_probs, dtype=torch.float)
        Ent = torch.tensor(storage.entropies, dtype=torch.float)
        policy_loss = -(L_p * A).mean()
        entropy_loss = Ent.mean()
        (policy_loss + self.entropy_beta*entropy_loss).backward()

        self.critic_optimizer.step()
        self.actor_optimizer.step()

    def train(self, episodes):
        # Initialize episodes counter, storage and initial state
        state = self.env.reset(train_mode=True)[
            self.brain_name].vector_observations[0]
        scores = []
        scores_window = deque(maxlen=100)
        storage = Storage(0, [
            "rewards",
            "action_values",
            "dones",
            "log_probs",
            "entropies",
            "advantages",
            "returns"
        ]
        )
        for i_ep in range(episodes):
            score = 0
            storage.reset()
            # Collect an episode
            while True:
                state = torch.from_numpy(state).float().\
                    to(self.device)

                # retrieve action
                mu, var = self.actor(state)
                dist = torch.distributions.Normal(
                    loc=mu,
                    scale=torch.exp(self.actor.logstd)
                )

                action = dist.sample()
                action = torch.clamp(action, -1, 1)
                log_prob = dist.log_prob(action).sum(-1).unsqueeze(-1)
                entropy = dist.entropy().sum(-1).unsqueeze(-1)

                # environment step based on action
                env_info = self.env.step(action.cpu().data.numpy())[
                    self.brain_name]
                next_state = env_info.vector_observations[0]  # next_state
                reward = env_info.rewards[0]
                done = env_info.local_done[0]

                # experience
                e = {
                    "rewards": torch.tensor(reward, dtype=torch.float),
                    "action_values": self.critic(state),
                    "dones": torch.tensor(done, dtype=torch.int),
                    "log_probs": log_prob,
                    "entropies": entropy
                }
                storage.add(e)

                if done:
                    state = self.env.reset(train_mode=True)[
                        self.brain_name].vector_observations[0]
                    break
                else:
                    state = next_state
                    score += reward

            storage.size = len(storage.rewards)
            storage.placeholder()

            # If our episode didn't end on
            # the last step we need to compute the value
            # for the last state

            if storage.dones[-1]:
                next_value = 0
            else:
                next_value = self.critic(torch.tensor(
                    state, dtype=torch.float)).detach().numpy()[0]

            # Calculate returns and advantages

            R = np.append(
                np.zeros_like(
                    storage.rewards
                ),
                [next_value],
                axis=0
            )

            for t in reversed(range(storage.size)):
                R[t] = storage.rewards[t] + self.gamma * \
                    R[t + 1] * (1 - storage.dones[t])

            for t in range(storage.size):
                storage.returns[t] = torch.tensor(R[t], dtype=torch.float).\
                    unsqueeze(-1)
                storage.advantages[t] = storage.returns[t] - \
                    storage.action_values[t]

            # Optimize Models
            self.optimize_models(storage)

            # Append scores
            scores_window.append(score)       # save most recent score
            scores.append(score)              # save most recent score

            print('\rEpisode {}\tAverage Score: {:.2f}'.format(
                i_ep, np.mean(scores_window)), end="")
            if i_ep % 100 == 0:
                print('\rEpisode {}\tAverage Score: {:.2f}'.format(
                    i_ep, np.mean(scores_window)))

            if np.mean(scores_window) >= 30.0:
                print('\nEnvironment solved in {:d} \
                    episodes!\tAverage Score: {:.2f}'.format(
                    i_ep-100, np.mean(scores_window)))
                torch.save(agent.actor_local.state_dict(),
                           'a2c_checkpoint.pth')
                break
