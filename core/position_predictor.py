from collections import deque
from functools import reduce
from math import pi
from math import sqrt

import gym
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from base.control import Control
from base.environment import Environment
from dataset.action_position_dataset import ActionPositionDataset
from nn.expert_policy import long_term_MPC, short_term_MPC
from nn.simple_action import SimpleNetwork
from utils.geo import to_relative_coords, to_absolute_coords
from utils.vis import plot_waypoints

N = 30
INF = float('inf')
MAX_steer = pi
MIN_steer = -pi
MAX_a = 10
MIN_a = -10
MAX_v = 100
MIN_v = 0


def measure_action_diff( pnts1, pnts2):
    return sqrt(sum([(a - b) ** 2 for a, b in zip(pnts1, pnts2)]))


class Test:
    def __init__(self, env: Environment, control_layer, position_layer=None):
        self.env = env
        self.dt = self.env.dt
        self.long_term_planning_length = self.env.long_term_planning_length
        self.short_term_planning_length = self.env.short_term_planning_length
        self.control_layer = control_layer
        self.long_term_mpc = long_term_MPC
        self.short_term_mpc = short_term_MPC
        self.position_layer = position_layer

    def run(self):
        self.position_layer.eval()
        self.env.reset()
        done = False
        curr_pos = self.env.get_car().get_position()
        prev_pos = (curr_pos.x - 10, curr_pos.y + 10)

        # states = deque(maxlen=N)
        # controls = deque(maxlen=N)
        cnt = 0

        with torch.no_grad():
            while not done:
                target_xs, target_ys = self.env.calc_long_term_targets()
                car_x, car_y, car_vx, car_vy, theta = self.env.observe()
                rel_long_term_waypoints = to_relative_coords(car_x, car_y, theta, target_xs, target_ys)

                control_traj_pred = self.control_layer(torch.FloatTensor(rel_long_term_waypoints).cuda()).detach().cpu().numpy()
                # control = Control(control_traj_pred[0], control_traj_pred[1], control_traj_pred[2])
                # states.append((car_x, car_y, car_vx, car_vy, theta))
                # controls.append(control_traj_pred)

                # if len(states) == N:
                #     car_x0, car_y0, car_vx0, car_vy0, theta0 = states[0]
                #
                #     s0 = [car_vx0, car_vy0]
                #     xs = [s[0] for s in states]
                #     ys = [s[1] for s in states]

                    # x_batch = [tuple(reduce(lambda x, y: x + y, [c.to_tuple() for c in controls])) + s0]

                # positions = to_relative_coords(car_x0, car_y0, theta0, xs, ys)
                pred_pos = self.position_layer(torch.FloatTensor(np.hstack((control_traj_pred, car_vx, car_vy))).cuda()).detach().cpu().numpy()
                short_term_target_xs = target_xs[:self.short_term_planning_length]
                short_term_target_ys = target_ys[:self.short_term_planning_length]
                pred_xs, pred_ys = pred_pos[:len(pred_pos) // 2], pred_pos[len(pred_pos) // 2:]
                pred_xs, pred_ys = to_absolute_coords(car_x, car_y, theta, short_term_target_xs, short_term_target_ys, False)
                pred_xs = pred_xs[:self.short_term_planning_length]
                pred_ys = pred_ys[:self.short_term_planning_length]

                mpc_control, _, _ = self.short_term_mpc(self.env.get_car(), short_term_target_xs, short_term_target_ys, self.dt)
                control_traj_mpc = np.hstack([mpc_control.to_tuple(), control_traj_pred[3:]])
                pred_pos_mpc = self.position_layer(torch.FloatTensor(np.hstack((control_traj_mpc, car_vx, car_vy))).cuda()).detach().cpu().numpy()

                target = target_xs + target_ys
                diff_nn = measure_action_diff(target, pred_pos)
                diff_mpc = measure_action_diff(target, pred_pos_mpc)

                nn_control = Control(control_traj_pred[0], control_traj_pred[1], control_traj_pred[2])

                if diff_nn > diff_mpc:
                    control = mpc_control
                    print('mpc', control.to_tuple())
                else:
                    control = nn_control
                    print('nn', control.to_tuple())
                # control = mpc_control

                # control, _, _ = self.long_term_mpc(self.env.get_car(), pred_xs, pred_ys, self.dt)

                # if not cnt % 25:
                #     plot_waypoints(pred_pos, pred_pos_mpc)
                cnt += 1
                    # dataset.record(list(controls), s0, positions)
                    # print("Dataset size: {}".format(cnt))
                    # cnt += 1

                _, _, done, _ = self.env.step(control)
                # _, _, done, _ = self.env.step(action)

                # Out of lane
                # curr_pos = self.env.get_car().get_position()
                # dist = (curr_pos.x - prev_pos[0]) ** 2 + (curr_pos.y - prev_pos[1]) ** 2
                # # print(dist)
                # if dist < 5 * 1e-4:  # Hyper parameter
                #     print("Out of boundary!")
                #     break
                # prev_pos = (curr_pos.x, curr_pos.y)

                self.env.render()


class ActionController:

    def __init__(self, env: Environment, lr=0.01):
        self.env = env
        self.dt = self.env.dt
        self.long_term_planning_length = self.env.long_term_planning_length
        self.expert = long_term_MPC

        self.action_layer = SimpleNetwork(self.long_term_planning_length * 3 + 2,
                                          self.long_term_planning_length * 2).cuda()
        self.optimizer = optim.Adam(self.action_layer.parameters(), lr=lr)

        self.init_dataset_size = 20000
        self.dataset_prime_size = 500
        self.M = 1
        self.waypoints_threshhold = 1e-3
        self.epoches = 10000

    def init_dataset(self, save=True):
        from collections import deque

        dataset = ActionPositionDataset()
        cnt = 0
        N = self.long_term_planning_length
        while dataset.size() < self.init_dataset_size:
            done = False
            self.env.reset()
            new_dataset = ActionPositionDataset()
            curr_pos = self.env.get_car().get_position()
            prev_pos = (curr_pos.x - 10, curr_pos.y + 10)
            out_of_boundary = False

            states = deque(maxlen=N)
            controls = deque(maxlen=N)

            while not out_of_boundary and not done and dataset.size() < self.init_dataset_size:
                target_xs, target_ys = self.env.calc_long_term_targets()
                control, _, _ = self.expert(self.env.car, target_xs, target_ys, self.dt)

                car_x, car_y, car_vx, car_vy, theta = self.env.observe()
                states.append((car_x, car_y, car_vx, car_vy, theta))
                controls.append(control)

                if len(states) == N:
                    car_x0, car_y0, car_vx0, car_vy0, theta0 = states[0]

                    s0 = (car_vx0, car_vy0)
                    xs = [s[0] for s in states]
                    ys = [s[1] for s in states]
                    positions = to_relative_coords(car_x0, car_y0, theta0, xs, ys)
                    control_traj = tuple(reduce(lambda x, y: x + y, [c.to_tuple() for c in controls]))

                    dataset.record(control_traj, s0, positions)
                    print("Dataset size: {}".format(cnt))
                    cnt += 1

                # Take action
                obs, _, done, _ = self.env.step(control)

                # Check if in lane
                curr_pos = self.env.get_car().get_position()
                dist = (curr_pos.x - prev_pos[0]) ** 2 + (curr_pos.y - prev_pos[1]) ** 2
                if dist < 1e-4:
                    out_of_boundary = True
                    print("Out of boundary!")
                prev_pos = (curr_pos.x, curr_pos.y)

            dataset += new_dataset

        if save:
            dataset.save()

        return dataset

    def _train(self, dataset: ActionPositionDataset):
        self.action_layer.train()
        epoch = 0
        while epoch < self.epoches:
            batch = dataset.fetch_random_batch()
            if not batch:
                break
            inputs, outputs = batch
            inputs = torch.FloatTensor(inputs).cuda()
            outputs = torch.FloatTensor(outputs).cuda()

            outputs_pred = self.action_layer(inputs)
            loss = F.mse_loss(outputs, outputs_pred)
            # if not epoch % 100:
            print("Epoch {}, loss: {}".format(epoch, loss))
            # print(outputs.detach().cpu().numpy())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            epoch += 1

    def _test(self):
        test = Test(self.env, self.action_layer)
        test.run()

    def run_dagger(self, dataset_path=None):

        # Build Dataset
        if dataset_path:
            dataset = ActionPositionDataset()
            dataset.load(dataset_path)
        else:
            dataset = self.init_dataset()
        print("Done init")

        round = 0
        while True:
            # Train NN with D
            self._train(dataset)
            self.save("../saved/action_pos/nn_policy_{}.h5".format(round))
            return

            # Run NN + label with LT-MPC
            dataset_prime = ActionPositionDataset()
            step = 0
            curr_pos = self.env.get_car().get_position()
            prev_pos = (curr_pos.x - 10, curr_pos.y + 10)
            print("Round {}".format(round))

            # Collect new data
            while dataset_prime.size() < self.dataset_prime_size:
                self.env.reset()
                done = False
                out_of_boundary = False
                while not out_of_boundary and not done and dataset_prime.size() < self.dataset_prime_size:
                    # Make input
                    long_term_xs, long_term_ys = self.env.calc_long_term_targets()

                    car_x, car_y, _, _, theta = self.env.observe()
                    rel_long_term_waypoints = to_relative_coords(car_x, car_y, theta, long_term_xs, long_term_ys)

                    # NN plan
                    nn_action = self.action_layer(
                        torch.FloatTensor(rel_long_term_waypoints).cuda()).detach().cpu().numpy()

                    # DAgger
                    if not step % self.M:
                        expert_action, expert_xs, expert_ys = self.expert(self.env.car, long_term_xs, long_term_ys,
                                                                          self.dt)

                        diff = measure_action_diff(expert_action.to_tuple(), nn_action.to_tuple())
                        if diff > self.waypoints_threshhold:
                            dataset_prime.record(nn_action, rel_long_term_waypoints, rel_long_term_waypoints)

                        if not dataset_prime.size() % 100:
                            print("nn action: {}, expert action: {}".format(nn_action, expert_action))
                            print("New item: {}, Difference: {}\n\n".format(dataset_prime.size(), diff))

                    _, _, done, _ = self.env.step(nn_action)

                    # Stop if out of boundary
                    curr_pos = self.env.get_car().get_position()
                    dist = (curr_pos.x - prev_pos[0]) ** 2 + (curr_pos.y - prev_pos[1]) ** 2
                    if dist < 5 * 1e-4:  # Hyper parameter
                        print("Out of boundary!")
                        break
                    prev_pos = (curr_pos.x, curr_pos.y)
                    step += 1

            round += 1
            self._test()

            self.save("./saved/action/nn_policy_{}.h5".format(round))
            print("Model saved\n\n" + "=" * 50)

            # add D' to D
            dataset += dataset_prime

    def save(self, save_path):
        torch.save(self.action_layer.state_dict(), save_path)


if __name__ == '__main__':
    env = gym.make('CarRacing-v0')
    env = Environment(env=env, FPS=50.0)

    model_dict = "../saved/action/nn_policy_0.h5"
    action_layer = SimpleNetwork(60, 90).cuda()
    action_layer.load_state_dict(torch.load(model_dict))

    model_dict = "../saved/action_pos/nn_policy_0.h5"
    position_layer = SimpleNetwork(92, 60).cuda()
    position_layer.load_state_dict(torch.load(model_dict))

    test = Test(env=env, position_layer=position_layer, control_layer=action_layer)
    test.run()

    # dagger = ActionController(env)
    # dagger.run_dagger("../cache/action_position_dataset.pkl")
    # dagger.run_dagger()

