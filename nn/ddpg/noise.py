import numpy as np
class GuassianNoise():

    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def add_noise(self, action: np.array):
        return action + np.random.normal(self.mu, self.sigma, size=action.shape)