""" Learn a nn using ddpg for the reach task"""
import copy
import time
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

import gym

from policy.ddpg.actor_critic import Actor, Critic
from policy.ddpg.noise import GuassianNoise
from policy.ddpg.replay_buffer import ReplayBuffer

np.random.seed(1000)

class DDPG():
    def __init__(
            self,
            env,
            action_dim,
            state_dim,
            critic_lr=3e-4,
            actor_lr=3e-4,
            gamma=0.99,
            batch_size=100,
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
        self.replay_buffer = ReplayBuffer(buffer_size=10000, init_length=1000, state_dim=state_dim,
                                          action_dim=action_dim, env=env)

    def weighSync(self, target_model, source_model, tau=0.001):
        for target_param, source_param in zip(target_model.parameters(), source_model.parameters()):
            target_param.data = source_param.data * tau + target_param.data * (1 - tau)

    def update(self):
        self.update_network()
        self.update_target_networks()

    # TODO: Complete the function
    def update_target_networks(self):
        """
        A function to update the target networks
        """
        self.weighSync(self.actor_target, self.actor)
        self.weighSync(self.critic_target, self.critic)

    # TODO: Complete the function
    def update_network(self):
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
        Q_next = self.critic_target.forward(next_states, next_actions.detach())
        Q_prime = rewards + undones.mul(self.gamma * Q_next)

        # Step 2: Calculate critic loss
        Q = self.critic.forward(states, actions)
        critic_loss = F.mse_loss(Q, Q_prime)

        # Actor loss
        policy_loss = -self.critic.forward(states, self.actor.forward(states)).mean()

        # update networks
        self.optimizer_actor.zero_grad()
        policy_loss.backward()
        self.optimizer_actor.step()

        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        self.optimizer_critic.step()

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
        Train the nn for the given number of iterations
        :param num_steps:The number of steps to train the nn for
        """
        returns = []

        step = 0
        episode = 0
        while step < num_steps:
            state = self.env.reset()
            episode_return = 0
            episode += 1

            while step < num_steps:
                action = self.actor.forward(
                    Variable(torch.from_numpy(state).cuda().float().unsqueeze(0))).detach().cpu().numpy()[0]
                action = noise.add_noise(action)
                new_state, reward, done, _ = self.env.step(action)
                self.replay_buffer.buffer_add(exp=(state, action, reward, new_state, 1.0 - done))

                state = new_state
                episode_return += reward

                if self.replay_buffer.is_warmup_done():
                    self.update()
                    returns.append((step, self.test()))

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

        ddpg_object = DDPG(
            env=env,
            state_dim=8,
            action_dim=2,
            critic_lr=1e-3,
            actor_lr=1e-3,
            gamma=0.99,
            batch_size=100,
        )
        # Train the nn
        noise = GuassianNoise(mu=0, sigma=0.1)
        returns = ddpg_object.train(num_steps=200000, noise=noise)

        xs = [r[0] for r in returns]
        ys = [r[1] for r in returns]
        plt.plot(xs, ys)
        plt.savefig('ddpg_return_seed_{}.png'.format(seed))
        plt.clf()

        # Evaluate the final nn
        state = env.reset()
        done = False
        while not done:
            action = \
            ddpg_object.actor(Variable(torch.from_numpy(state).float().unsqueeze(0)).cuda()).detach().cpu().numpy()[0]
            next_state, r, done, _ = env.step(action)
            env.render()
            time.sleep(0.1)
            state = next_state
