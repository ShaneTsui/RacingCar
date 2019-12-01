from math import atan2
import numpy as np

from base.action import Action


class Car(object):

    def __init__(self, car, a, steer):
        self.car = car
        self.wheel = car.wheels[0]

        # Previous control
        self.accel = a
        self.steer = steer

    def get_velocity(self):
        vx, vy = self.wheel.linearVelocity * 1
        return np.linalg.norm((vx, vy))

    def get_wheel(self):
        return self.wheel

    def get_position(self):
        return self.car.hull.position

    def get_car_state(self):
        x, y = self.car.hull.position
        vx, vy = self.wheel.linearVelocity * 1.
        v = np.linalg.norm((vx, vy))
        theta = atan2(vy, vx)
        return x, y, v, theta, self.accel, self.steer

    def take_control(self, control: Action):
        self.accel = control.accel
        self.steer = control.steer