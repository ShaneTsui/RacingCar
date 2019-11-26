from math import atan2
import numpy as np

class Car(object):

    def __init__(self, car, a, steer):

        self.car = car
        self.wheel = car.wheels[0]

        # Previous control
        self.accel = a
        self.steer = steer

    def get_car_state(self):
        x, y = self.car.hull.position
        vx, vy = self.wheel.linearVelocity * 1
        v = np.linalg.norm((vx, vy))
        theta = atan2(vy, vx)
        return x, y, v, theta, self.accel, self.steer

    def take_control(self, accel, steer):
        self.accel = accel
        self.steer = steer