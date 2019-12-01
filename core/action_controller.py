from math import pi
from math import sqrt

import gym
import torch
import torch.nn.functional as F
import torch.optim as optim

from base.action import Action
from base.environment import Environment
from dataset.action_dataset import ActionDataset
from nn.expert_policy import long_term_MPC
from nn.simple_action import ActionNetwork
from utils.geo import to_relative_coords

N = 30
INF = float('inf')
MAX_steer = pi
MIN_steer = -pi
MAX_a = 10
MIN_a = -10
MAX_v = 100
MIN_v = 0


class Test:
    def __init__(self, env: Environment, action_layer=None):
        self.env = env
        self.dt = self.env.dt
        self.long_term_planning_length = self.env.long_term_planning_length
        self.action_layer = action_layer
        self.expert = long_term_MPC

    def run(self):
        self.action_layer.eval()
        self.env.reset()
        done = False
        curr_pos = self.env.get_car().get_position()
        prev_pos = (curr_pos.x - 10, curr_pos.y + 10)
        with torch.no_grad():
            while not done:
                long_term_xs, long_term_ys = self.env.calc_long_term_targets()
                car_x, car_y, _, _, theta = self.env.observe()
                rel_long_term_waypoints = to_relative_coords(car_x, car_y, theta, long_term_xs, long_term_ys)
                action = self.action_layer(torch.FloatTensor(rel_long_term_waypoints).cuda()).detach().cpu().numpy()

                _, _, done, _ = self.env.step(Action(action[0], action[1], action[2]))

                curr_pos = self.env.get_car().get_position()
                dist = (curr_pos.x - prev_pos[0]) ** 2 + (curr_pos.y - prev_pos[1]) ** 2
                print(dist)
                if dist < 5 * 1e-4:  # Hyper parameter
                    print("Out of boundary!")
                    break
                prev_pos = (curr_pos.x, curr_pos.y)

                self.env.render()


class ActionController:

    def __init__(self, env: Environment, lr=0.01):
        self.env = env
        self.dt = self.env.dt
        self.long_term_planning_length = self.env.long_term_planning_length
        self.expert = long_term_MPC

        self.action_layer = ActionNetwork(self.long_term_planning_length * 2, 3).cuda()
        self.optimizer = optim.Adam(self.action_layer.parameters(), lr=lr)

        self.init_dataset_size = 20000
        self.dataset_prime_size = 500
        self.M = 1
        self.waypoints_threshhold = 1e-3
        self.epoches = 2000

    def init_dataset(self, save=True):
        dataset = ActionDataset()
        cnt = 0
        while dataset.size() < self.init_dataset_size:
            done = False
            self.env.reset()
            new_dataset = ActionDataset()
            curr_pos = self.env.get_car().get_position()
            prev_pos = (curr_pos.x - 10, curr_pos.y + 10)
            out_of_boundary = False

            while not out_of_boundary and not done and dataset.size() < self.init_dataset_size:
                target_xs, target_ys = self.env.calc_long_term_targets()
                action, expert_long_term_xs, expert_long_term_ys = self.expert(self.env.car, target_xs, target_ys,
                                                                               self.dt)

                car_x, car_y, _, _, theta = self.env.observe()
                rel_targets = to_relative_coords(car_x, car_y, theta, target_xs, target_ys)
                rel_long_term_waypoints = to_relative_coords(car_x, car_y, theta, expert_long_term_xs,
                                                             expert_long_term_ys)
                new_dataset.record(action, rel_targets, rel_long_term_waypoints)

                print("Dataset size: {}".format(cnt))
                cnt += 1

                # Take action
                obs, _, done, _ = self.env.step(action)

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

    def _measure_action_diff(self, pnts1, pnts2):
        return sqrt(sum([(a - b) ** 2 for a, b in zip(pnts1, pnts2)]))

    def _train(self, dataset: ActionDataset):
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
            dataset = ActionDataset()
            dataset.load(dataset_path)
        else:
            dataset = self.init_dataset()
        print("Done init")

        round = 0
        while True:
            # Train Policy NN with D
            self._train(dataset)
            self.save("../saved/action/nn_policy_{}.h5".format(round))
            return

            # Run Policy NN + label with LT-MPC
            dataset_prime = ActionDataset()
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

                        diff = self._measure_action_diff(expert_action.to_tuple(), nn_action.to_tuple())
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
    action_layer = ActionNetwork(60, 3).cuda()
    action_layer.load_state_dict(torch.load(model_dict))

    test = Test(env, action_layer)
    test.run()

    # dagger = ActionController(env)
    # dagger.run_dagger('../cache/action_dataset.pkl')
    # dagger.run_dagger("action_dataset.pkl")
