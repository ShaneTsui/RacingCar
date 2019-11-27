class Dataset:

    def __init__(self, batch_size=128):
        self.inputs = []
        self.outputs = []
        self.batch_size = batch_size
        self.idx = 0

    def record(self, input, output):
        self.inputs.append(input)
        self.outputs.append(output)

    def fetch(self, batch_size):
        raise NotImplementedError()

    def size(self):
        return len(self.inputs)

    def get_batch(self):
        if self.idx >= self.size():
            self.idx = 0
            return None
        x_batch = self.inputs[self.idx: self.idx + self.batch_size]
        y_batch = self.outputs[self.idx: self.idx + self.batch_size]
        self.idx += self.batch_size
        return x_batch, y_batch


    def __add__(self, other):
        dataset = Dataset()
        dataset.inputs = self.inputs + other.inputs
        dataset.outputs = self.outputs + other.outputs
        return dataset