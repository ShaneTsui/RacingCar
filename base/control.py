class Control:

    def __init__(self, steer, accel, brk):
        self.steer = steer
        self.accel = accel
        self.brk = brk

    def to_tuple(self):
        return self.steer, self.accel, self.brk

    def __str__(self):
        return "acc: {} , steer: {}, break: {}".format(self.accel, self.steer, self.brk)