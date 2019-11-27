from math import atan2

import numpy as np

from core.car import Car
from core.action import Action
# from utils import singleton


# @singleton
class Environment:

    def __init__(self, env, FPS=50.0):
        self.env = env
        self.env.reset()
        self.car = Car(env.unwrapped.car, 10., 0.)
        self.waypoints = env.unwrapped.track
        self.dt = 1 / FPS
        self.long_term_planning_length = 30
        self.short_term_planning_length = 5

    def render(self):
        self.env.render()

    def observe(self):
        x, y = self.car.get_position()
        vx, vy = self.car.get_wheel().linearVelocity * 1
        theta = atan2(vy, vx)
        return x, y, vx, vy, theta

    def step(self, control: Action):
        obs = self.observe()
        _, reward, done, _ = self.env.step(control.to_tuple())
        self.car.take_control(control)
        return obs, reward, done

    def reset(self):
        state = self.env.reset()
        self.car = Car(self.env.unwrapped.car, 10, 0)
        return state

    def calc_long_term_targets(self):
        curr_waypoint_idx = self.env.unwrapped.tile_visited_count
        curr_waypoint_idx = curr_waypoint_idx % len(self.env.unwrapped.track)

        pos = self.car.get_position()
        desired_v = 80
        dist_travel = desired_v * self.dt

        def get_point(start, end, d_to_go):
            x0, y0 = start
            x1, y1 = end
            dy = y1 - y0
            dx = x1 - x0
            d = np.linalg.norm((dx, dy))

            x = x0 + d_to_go * dx / d
            y = y0 + d_to_go * dy / d

            return np.array((x, y))

        cur_pos = np.array(pos)
        curr_waypoint_idx = curr_waypoint_idx % len(self.waypoints)
        cur_target = np.array(self.waypoints[curr_waypoint_idx][2:4])

        result = [pos]
        for i in range(self.long_term_planning_length - 1):
            remain_dist = np.linalg.norm(cur_target - cur_pos) - dist_travel
            if remain_dist > 0:
                p = get_point(cur_pos, cur_target, dist_travel)
                result.append(p)
                cur_pos = p
            else:
                # must ensure distance between 2 target points larger than dist_travel
                cur_pos = cur_target
                curr_waypoint_idx = (curr_waypoint_idx + 1) % len(self.waypoints)
                cur_target = np.array(self.waypoints[curr_waypoint_idx][2:4])

                p = get_point(cur_pos, cur_target, - remain_dist)
                result.append(p)
                cur_pos = p

        # xs = result['x'][0:self.long_term_planning_length].elements()
        # ys = result['x'][self.long_term_planning_length:2 * self.long_term_planning_length].elements()

        xs = [i[0] for i in result]
        ys = [i[1] for i in result]

        return xs, ys