import pickle
import random


class Dataset:

    def __init__(self, batch_size=512):
        self.inputs = []
        self.outputs = []
        self.batch_size = batch_size
        self.idx = 0

    def load(self, save_path):
        with open(save_path, 'rb') as f:
            d = pickle.load(f)
            self.inputs += d['inputs']
            self.outputs += d['outputs']

    def save(self, save_path='./action_dataset.pkl'):
        with open(save_path, 'wb') as f:
            pickle.dump({"inputs": self.inputs, "outputs": self.outputs}, f)

    def record(self, input, output):
        self.inputs.append(input)
        self.outputs.append(output)

    def size(self):
        return len(self.inputs)

    def fetch_random_batch(self):
        self.rand_idx = [i for i in range(self.size())]
        idxes = random.choices(self.rand_idx, k=self.batch_size)

        x_batch = [self.inputs[i] for i in idxes]
        y_batch = [self.outputs[i] for i in idxes]

        return x_batch, y_batch

    def __add__(self, other):
        dataset = Dataset()
        dataset.inputs = self.inputs + other.inputs
        dataset.outputs = self.outputs + other.outputs
        return dataset

if __name__ == '__main__':
    dataset = Dataset()

    dataset.load('../action_dataset.pkl')
    from utils.vis import plot_waypoints

    long_term, short_term = dataset.fetch_random_batch()

    for i in range(100):
        plot_waypoints(long_term[i], short_term[i])
