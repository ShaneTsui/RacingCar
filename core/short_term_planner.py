from math import pi
from math import sqrt

import gym
import torch
import torch.nn.functional as F
import torch.optim as optim

from dataset.dataset import Dataset
from base.environment import Environment
from nn.expert_policy import short_term_MPC, long_term_MPC
from nn.simple_policy import PolicyNetwork
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

class Test:
    def __init__(self, env: Environment, policy_layer=None):
        self.env = env
        self.dt = self.env.dt
        self.long_term_planning_length = self.env.long_term_planning_length
        self.short_term_planning_length = self.env.short_term_planning_length
        self.policy_layer = policy_layer
        self.expert = long_term_MPC
        self.execution_layer = short_term_MPC

    def run(self):
        self.policy_layer.eval()
        self.env.reset()
        done = False
        curr_pos = self.env.get_car().get_position()
        prev_pos = (curr_pos.x - 10, curr_pos.y + 10)
        with torch.no_grad():
            while not done:
                long_term_xs, long_term_ys = self.env.calc_long_term_targets()

                rel_long_term_waypoints = to_relative_coords(long_term_xs, long_term_ys)
                rel_nn_planned_waypoints = self.policy_layer(torch.FloatTensor(rel_long_term_waypoints).cuda())
                rel_nn_planned_waypoints = rel_nn_planned_waypoints.detach().cpu().numpy()
                abs_nn_planned_xs, abs_nn_planned_ys = to_absolute_coords(
                    rel_nn_planned_waypoints[:self.short_term_planning_length], \
                    rel_nn_planned_waypoints[self.short_term_planning_length: 2 * self.short_term_planning_length],
                    combine=False)

                action = self.execution_layer(self.env.car, abs_nn_planned_xs, abs_nn_planned_ys, self.dt)
                print(action)
                _, _, done, _ = self.env.step(action)

                curr_pos = self.env.get_car().get_position()
                dist = (curr_pos.x - prev_pos[0]) ** 2 + (curr_pos.y - prev_pos[1]) ** 2
                print(dist)
                if dist < 5 * 1e-4:  # Hyper parameter
                    print("Out of boundary!")
                    break
                prev_pos = (curr_pos.x, curr_pos.y)

                self.env.render()


class ShortTermPlanner:

    def __init__(self, env: Environment, lr=0.01):
        self.env = env
        self.expert = long_term_MPC
        self.dt = self.env.dt
        self.long_term_planning_length = self.env.long_term_planning_length
        self.short_term_planning_length = self.env.short_term_planning_length
        self.execution_layer = short_term_MPC

        self.policy_layer = PolicyNetwork(self.long_term_planning_length * 2,
                                          self.short_term_planning_length * 2).cuda()
        self.optimizer = optim.Adam(self.policy_layer.parameters(), lr=lr)

        self.init_dataset_size = 20000
        self.dataset_prime_size = 500
        self.M = 1
        self.waypoints_threshhold = 1e-3
        self.epoches = 10000


    def init_dataset(self, save=True):
        dataset = Dataset()
        cnt = 0
        while dataset.size() < self.init_dataset_size:
            done = False
            self.env.reset()
            new_dataset = Dataset()
            curr_pos = self.env.get_car().get_position()
            prev_pos = (curr_pos.x - 10, curr_pos.y + 10)
            out_of_boundary = False

            while not out_of_boundary and not done and dataset.size() < self.init_dataset_size:
                long_term_xs, long_term_ys = self.env.calc_long_term_targets()

                _, expert_long_term_xs, expert_long_term_ys = self.expert(self.env.car, long_term_xs, long_term_ys, self.dt)
                rel_long_term_waypoints = to_relative_coords(long_term_xs, long_term_ys)
                rel_short_term_waypoints = to_relative_coords(expert_long_term_xs, expert_long_term_ys)

                if cnt % 20 == 0:
                    # plot_waypoints(rel_long_term_waypoints, rel_short_term_waypoints)
                    plot_waypoints(long_term_xs + long_term_ys, expert_long_term_xs + expert_long_term_ys)
                new_dataset.record(rel_long_term_waypoints, rel_short_term_waypoints)
                print("Dataset size: {}".format(cnt))
                cnt += 1

                # Take action
                action = self.execution_layer(self.env.car, expert_long_term_xs[:self.short_term_planning_length],
                                              expert_long_term_ys[:self.short_term_planning_length], self.dt)
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

    def _measure_waypoints_diff(self, pnts1, pnts2):
        return sqrt(sum([(a - b) ** 2 for a, b in zip(pnts1, pnts2)]))

    def _train(self, dataset: Dataset):
        self.policy_layer.train()
        epoch = 0
        while epoch < self.epoches:
            batch = dataset.fetch_random_batch()
            if not batch:
                break
            inputs, outputs = batch
            inputs = torch.FloatTensor(inputs).cuda()
            outputs = torch.FloatTensor(outputs).cuda()

            outputs_pred = self.policy_layer(inputs)
            loss = F.mse_loss(outputs, outputs_pred)
            if not epoch % 100:
                print("Epoch {}, loss: {}".format(epoch, loss))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            epoch += 1

    def _test(self):
        test = Test(self.env, self.policy_layer)
        test.run()

    def run_dagger(self, dataset_path=None):

        # Build Dataset
        if dataset_path:
            dataset = Dataset()
            dataset.load(dataset_path)
        else:
            dataset = self.init_dataset()
        print("Done init")

        round = 0
        while True:
            # Train Policy NN with D
            self._train(dataset)

            # Run Policy NN + label with LT-MPC
            dataset_prime = Dataset()
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
                    rel_long_term_waypoints = to_relative_coords(long_term_xs, long_term_ys)

                    # NN plan
                    rel_nn_short_term_waypoints = self.policy_layer(torch.FloatTensor(rel_long_term_waypoints).cuda()).detach().cpu().numpy()

                    # DAgger
                    if not step % self.M:
                        _, expert_xs, expert_ys = self.expert(self.env.car, long_term_xs, long_term_ys, self.dt)
                        rel_expert_short_term_waypoints = to_relative_coords(expert_xs[:self.short_term_planning_length], expert_ys[:self.short_term_planning_length])

                        diff = self._measure_waypoints_diff(rel_nn_short_term_waypoints, rel_expert_short_term_waypoints)
                        if diff > self.waypoints_threshhold:
                            dataset_prime.record(rel_long_term_waypoints, rel_expert_short_term_waypoints)

                        if not dataset_prime.size() % 100:
                            print("rel_nn_short_term_waypoints", rel_nn_short_term_waypoints)
                            print("rel_expert_short_term_waypoints", rel_expert_short_term_waypoints)
                            print("New item: {}, Difference: {}\n\n".format(dataset_prime.size(), diff))

                    abs_nn_planned_xs, abs_nn_planned_ys = to_absolute_coords(
                        rel_nn_short_term_waypoints[:self.short_term_planning_length], \
                        rel_nn_short_term_waypoints[self.short_term_planning_length: 2 * self.short_term_planning_length],
                        combine=False)

                    action = self.execution_layer(self.env.car, abs_nn_planned_xs, abs_nn_planned_ys, self.dt)
                    _, _, done, _ = self.env.step(action)

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

            self.save("./saved/nn_policy_{}.h5".format(round))
            print("Model saved\n\n" + "=" * 50)

            # add D' to D
            dataset += dataset_prime

    def save(self, save_path):
        torch.save(self.policy_layer.state_dict(), save_path)

    def load(self):
        pass

    def policy(self):
        pass


if __name__ == '__main__':
    env = gym.make('CarRacing-v0')
    env = Environment(env=env, FPS=50.0)

    # test = Test(env)
    # test.run()

    dagger = ShortTermPlanner(env)
    dagger.run_dagger()
    # dagger.run_dagger("action_dataset.pkl")
