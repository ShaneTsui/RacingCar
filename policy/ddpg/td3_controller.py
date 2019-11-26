""" Learn a policy using ddpg for the reach task"""
import copy
import time
import random
import collections
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F


import gym
import pybullet
import pybulletgym.envs

from policy.ddpg.noise import GuassianNoise

np.random.seed(1000)


# TODO: A function to soft update target networks
def weighSync(target_model, source_model, tau=0.001):
    for target_param, source_param in zip(target_model.parameters(), source_model.parameters()):
        target_param.data = source_param.data * tau + target_param.data * (1 - tau)

# TODO: Write the ReplayBuffer
class ReplayBuffer():
    def __init__(self, buffer_size, init_length, state_dim, action_dim, env):
        """
        A function to initialize the replay buffer.

        param: init_length : Initial number of transitions to collect
        param: state_dim : Size of the state space
        param: action_dim : Size of the action space
        param: env : gym environment object
        """
        self.buffer = collections.deque(maxlen=buffer_size)
        self.state_dim = state_dim
        self.init_length = init_length
        self.action_dim = action_dim
        self.env = env

    # TODO: Complete the function
    def buffer_add(self, exp):
        """
        A function to add a dictionary to the buffer
        param: exp : A tuple consisting of (state, action, reward, next_state, undone)
        """
        self.buffer.append(exp)

    #TODO: Complete the function
    def buffer_sample(self, N):
        """
        A function to sample N points from the buffer
        param: N : Number of samples to obtain from the buffer
        """
        samples = random.sample(self.buffer, N)

        states, actions, rewards, next_states, undones = [], [], [], [], []
        for state, action, reward, next_state, undone in samples:
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            undones.append(undone)

        return states, actions, rewards, next_states, undones

    def is_warmup_done(self):
        return len(self.buffer) > self.init_length

# TODO: Define an Actor
class Actor(nn.Module):
    #TODO: Complete the function
    def __init__(self, state_dim, action_dim):
        """
        Initialize the network
        param: state_dim : Size of the state space
        param: action_dim: Size of the action space
        """
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(state_dim, 256)
        self.linear2 = nn.Linear(256, 256)
        self.linear3 = nn.Linear(256, action_dim)

    #TODO: Complete the function
    def forward(self, state):
        """
        Define the forward pass
        param: state: The state of the environment
        """
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = torch.tanh(self.linear3(x))
        return x


# TODO: Define the Critic
class Critic(nn.Module):
    # TODO: Complete the function
    def __init__(self, state_dim, action_dim):
        """
        Initialize the critic
        param: state_dim : Size of the state space
        param: action_dim : Size of the action space
        """
        super(Critic, self).__init__()
        # Q1
        self.linear1 = nn.Linear(state_dim + action_dim, 256)
        self.linear2 = nn.Linear(256, 256)
        self.linear3 = nn.Linear(256, 1)

        # Q2
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    # TODO: Complete the function
    def forward(self, state, action):
        """
        Define the forward pass of the critic
        """
        input = torch.cat([state, action], 1)

        q1 = F.relu(self.linear1(input))
        q1 = F.relu(self.linear2(q1))
        q1 = self.linear3(q1)

        q2 = F.relu(self.l4(input))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)

        return q1, q2

    def Q1(self, state, action):
        input = torch.cat([state, action], 1)

        q1 = F.relu(self.linear1(input))
        q1 = F.relu(self.linear2(q1))
        q1 = self.linear3(q1)
        return q1

# TODO: Implement a ddpg class
class TD3():
    def __init__(
            self,
            env,
            action_dim,
            state_dim,
            critic_lr=3e-4,
            actor_lr=3e-4,
            gamma=0.99,
            batch_size=200,
            policy_freq=3
    ):
        """
        param: env: An gym environment
        param: action_dim: Size of action space
        param: state_dim: Size of state space
        param: critic_lr: Learning rate of the critic
        param: actor_lr: Learning rate of the actor
        param: gamma: The discount factor
        param: batch_size: The batch size for training
        """
        self.gamma = gamma
        self.batch_size = batch_size
        self.env = env
        self.test_env = copy.deepcopy(env)
        self.policy_freq =  policy_freq

        # TODO: Create a actor and actor_target
        self.actor = Actor(state_dim, action_dim).cuda()
        self.actor_target = Actor(state_dim, action_dim).cuda()
        self.critic = Critic(state_dim, action_dim).cuda()
        self.critic_target = Critic(state_dim, action_dim).cuda()

        # TODO: Make sure that both networks have the same initial weights
        for actor_target_param, actor_param in zip(self.actor_target.parameters(), self.actor.parameters()):
            actor_target_param.data = actor_param.data.clone()
        
        for critic_target_param, critic_param in zip(self.critic_target.parameters(), self.critic.parameters()):
            critic_target_param.data = critic_param.data.clone()

        # TODO: Define the optimizer for the actor
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=actor_lr)

        # TODO: Define the optimizer for the critic
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=critic_lr)

        # TODO: define a replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size=10000, init_length=1000, state_dim=state_dim, action_dim=action_dim, env=env)

    # TODO: Complete the function
    def update_target_networks(self):
        """
        A function to update the target networks
        """
        weighSync(self.actor_target, self.actor)
        weighSync(self.critic_target, self.critic)

    # TODO: Complete the function
    def update(self, step):
        """
        A function to update the function just once
        """
        # Obtain data from replay buffer & translate to tensor
        states, actions, rewards, next_states, undones = self.replay_buffer.buffer_sample(self.batch_size)
        states = torch.FloatTensor(states).cuda()
        actions = torch.FloatTensor(actions).cuda()
        rewards = torch.FloatTensor(rewards).unsqueeze(1).cuda()
        next_states = torch.FloatTensor(next_states).cuda()
        undones = torch.FloatTensor(undones).unsqueeze(1).cuda()

        # Critic loss    
        # Step 1: Calculate Q prime using target networks
        next_actions = self.actor_target.forward(next_states)

        # Compute the target Q value
        target_Q1, target_Q2 = self.critic_target(next_states, next_actions)
        target_Q = torch.min(target_Q1, target_Q2)
        target_Q = rewards + undones.mul(self.gamma * target_Q)

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(states, actions)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        self.optimizer_critic.step()

        # Delayed policy updates
        if step % self.policy_freq == 0:

            # Compute actor loss
            policy_loss = -self.critic.Q1(states, self.actor(states)).mean()

            # Optimize the actor
            self.optimizer_actor.zero_grad()
            policy_loss.backward()
            self.optimizer_actor.step()

            self.update_target_networks()

    def test(self):
        state = self.test_env.reset()
        done = False
        ret = 0
        while not done:
            action = self.actor(Variable(torch.from_numpy(state).float().unsqueeze(0)).cuda()).detach().cpu().numpy()[0]
            next_state, r, done, _ = self.test_env.step(action)
            ret += r
            state = next_state
        return ret

    # TODO: Complete the function
    def train(self, num_steps, noise):
        """
        Train the policy for the given number of iterations
        :param num_steps:The number of steps to train the policy for
        """
        returns = []

        step = 0
        episode = 0
        while step < num_steps:
            state = self.env.reset()
            episode_return = 0
            episode += 1
            
            while step < num_steps:
                action = self.actor.forward(Variable(torch.from_numpy(state).cuda().float().unsqueeze(0))).detach().cpu().numpy()[0]
                action = noise.add_noise(action)

                next_state, reward, done, _ = self.env.step(action)
                self.replay_buffer.buffer_add(exp=(state, action, reward, next_state, 1.0 - done))

                state = next_state
                episode_return += reward

                if self.replay_buffer.is_warmup_done():
                    self.update(step)
                    ret = self.test()
                    returns.append((step, ret))
                    print("Return: {}".format(ret))

                if done:
                    break
                step += 1

        return returns

if __name__ == "__main__":

    for seed in [0, 7, 22]:
        env = gym.make("modified_gym_env:ReacherPyBulletEnv-v1", rand_init=False)
        random.seed(seed)
        env.seed(seed)
        torch.manual_seed(seed)

        td3_object = TD3(
            env=env,
            state_dim=8,
            action_dim=2,
            critic_lr=1e-3,
            actor_lr=1e-3,
            gamma=0.99,
            batch_size=100,
            policy_freq=2
        )
        # Train the policy
        noise = GuassianNoise(mu=0, sigma=0.1)
        returns = td3_object.train(num_steps=200000, noise=noise)

        xs = [r[0] for r in returns]
        ys = [r[1] for r in returns]
        plt.plot(xs, ys)
        plt.savefig('td3_return_seed_{}.png'.format(seed))
        plt.clf()

        # Evaluate the final policy
        state = env.reset()
        done = False
        while not done:
            action = td3_object.actor(Variable(torch.from_numpy(state).float().unsqueeze(0)).cuda()).detach().cpu().numpy()[0]
            next_state, r, done, _ = env.step(action)
            env.render()
            time.sleep(0.1)
            state = next_state
