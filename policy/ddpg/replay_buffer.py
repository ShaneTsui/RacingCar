import collections
import random


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