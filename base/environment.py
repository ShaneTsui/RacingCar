from math import atan2

import numpy as np

from base.control import Control
from base.car import Car


class Environment:

    def __init__(self, env, FPS=50.0):
        self.env = env
        self.env.reset()
        self.car = Car(env.unwrapped.car, 10., 0.)
        self.track = env.unwrapped.track
        self.dt = 1 / FPS
        self.long_term_planning_length = 30
        self.short_term_planning_length = 10

    def render(self):
        self.env.render()

    def observe(self):
        x, y = self.car.get_position()
        vx, vy = self.car.get_wheel().linearVelocity * 1
        theta = atan2(vy, vx)
        return x, y, vx, vy, theta

    def step(self, control: Control):
        obs = self.observe()
        _, reward, done, info = self.env.step(control.to_tuple())
        self.car.take_control(control)
        return obs, reward, done, info

    def seed(self, seed_val):
        self.env.seed(seed_val)

    def reset(self):
        state = self.env.reset()
        self.car = Car(self.env.unwrapped.car, 10, 0)
        return state

    def get_car(self):
        return self.car

    def get_current_waypoint_index(self):
        ind = self.env.unwrapped.tile_visited_count
        ind = ind % len(self.get_track())
        return ind

    def get_track(self):
        return self.env.unwrapped.track

    def calc_long_term_targets(self):
        ind = self.get_current_waypoint_index()
        track = self.get_track()

        pos = self.car.get_position()

        desired_v = 60
        dist_travel = desired_v * self.dt

        def get_point(start, end, d_to_go):
            x0, y0 = start
            x1, y1 = end
            dy = y1 - y0
            dx = x1 - x0
            d = np.linalg.norm((dx, dy))

            x = x0 + d_to_go * dx / d
            y = y0 + d_to_go * dy / d

            return x, y

        cur_pos = np.array(pos)
        cur_target = np.array(track[ind][2:4])

        # result = [pos]
        xs, ys = [pos[0]], [pos[1]]
        for i in range(self.long_term_planning_length - 1):
            remain_dist = np.linalg.norm(cur_target - cur_pos) - dist_travel
            if remain_dist > 0:
                p = get_point(cur_pos, cur_target, dist_travel)
                # result.append(p)
                xs.append(p[0])
                ys.append(p[1])
                cur_pos = p
            else:
                # must ensure distance between 2 target points larger than dist_travel
                cur_pos = cur_target
                ind = (ind + 1) % len(track)
                cur_target = np.array(track[ind][2:4])

                p = get_point(cur_pos, cur_target, -remain_dist)
                xs.append(p[0])
                ys.append(p[1])
                cur_pos = p

        return xs, ys
