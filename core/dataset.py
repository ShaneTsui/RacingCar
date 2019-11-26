class Dataset:

    def __init__(self):
        self.waypoint_xs = []
        self.waypoint_ys = []
        self.vs = []
        self.accels = []
        self.steers = []

    def record(self, x, y, v, a, s):
        self.waypoint_xs.append(x)
        self.waypoint_ys.append(y)
        self.vs.append(v)
        self.accels.append(a)
        self.steers.append(s)

    def fetch(self, batch_size):
        raise NotImplementedError()