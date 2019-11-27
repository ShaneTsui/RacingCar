class Dataset:

    def __init__(self):
        self.inputs = []
        self.outputs = []


    def record(self, input, output):
        self.inputs.append(input)
        self.outputs.append(output)

    def fetch(self, batch_size):
        raise NotImplementedError()

    def size(self):
        return len(self.inputs)

    def __add__(self, other):
        dataset = Dataset()
        dataset.inputs = self.inputs + other.inputs
        dataset.outputs = self.outputs + other.inputs
        return dataset