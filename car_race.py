import gym
from math import sqrt, sin, cos, tan, atan, atan2, pi
from casadi import *
from time import time


lr = 1.4
lf = 1.4
FPS = 50.0

INF = float('inf')
MAX_steer = pi
MIN_steer = -pi
MAX_a = 10
MIN_a = -10
MAX_v = 100
MIN_v = 0


class Car(object):

    def __init__(self, obs, control):
        x, y, theta, dx, dy, dtheta, target = obs
        self.x = x
        self.y = y
        self.theta = theta
        self.v = np.linalg.norm((dx, dy))

        a, steer = control
        self.steer = steer
        self.a = a


# tracking problem solver
def long_term_MPC(ego_car, route, dt, N):
    # initial state
    x_init = [ego_car.x] * N
    y_init = [ego_car.y] * N
    theta_init = [ego_car.theta] * N
    v_init = [ego_car.v] * N
    steer_init = [ego_car.steer] * (N - 1)
    a_init = [MAX_a] * (N - 1)
    initial_value = x_init + y_init + theta_init + v_init +\
                    steer_init + a_init

    # vars to be optimize
    x = SX.sym('x', N)
    y = SX.sym('y', N)
    theta = SX.sym('theta', N)
    v = SX.sym('v', N)
    steer = SX.sym('steer', N - 1)
    a = SX.sym('a', N - 1)
    all_vars = vertcat(x, y, theta, v, steer, a)

    # vars upper bound
    ub_constrains_x = np.array([ego_car.x]+[INF] * (N-1))
    ub_constrains_y = np.array([ego_car.y]+[INF] * (N-1))
    ub_constrains_theta = np.array([ego_car.theta]+[INF] * (N-1))
    ub_constrains_v = np.array([ego_car.v]+[MAX_v] * (N-1))
    ub_constrains_steer = np.array([ego_car.steer]+[MAX_steer] * (N-2))
    ub_constrains_a = np.array([ego_car.a]+[MAX_a] * (N-2))
    ub_constrains_vars = np.hstack([ub_constrains_x, ub_constrains_y,
                                    ub_constrains_theta, ub_constrains_v,
                                    ub_constrains_steer, ub_constrains_a])
    # vars lower bound
    lb_constrains_x = np.array([ego_car.x]+[-INF] * (N-1))
    lb_constrains_y = np.array([ego_car.y]+[-INF] * (N-1))
    lb_constrains_theta = np.array([ego_car.theta]+[-INF] * (N-1))
    lb_constrains_v = np.array([ego_car.v]+[-MIN_v] * (N-1))
    lb_constrains_steer = np.array([ego_car.steer]+[MIN_steer] * (N-2))
    lb_constrains_a = np.array([ego_car.a]+[MIN_a] * (N-2))
    lb_constrains_vars = np.hstack([lb_constrains_x, lb_constrains_y,
                                    lb_constrains_theta, lb_constrains_v,
                                    lb_constrains_steer, lb_constrains_a])

    # define constrain function g (variables update equation)
    x_constrain = SX.sym('x_constrain', N - 1)
    y_constrain = SX.sym('y_constrain', N - 1)
    theta_constrain = SX.sym('theta_constrain', N - 1)
    v_constrain = SX.sym('v_constrain', N - 1)

    SCALE = 0.002
    for i in range(N-1):
        theta_diff = atan(tan(steer[i]) / 2) * v[i] * dt * SCALE
        # theta_diff = steer[i] * dt

        x_constrain[i] = x[i + 1] - (x[i] + v[i] * dt * np.cos(theta[i]))
        y_constrain[i] = y[i + 1] - (y[i] + v[i] * dt * np.sin(theta[i]))
        theta_constrain[i] = theta[i + 1] - (theta[i] - theta_diff)
        v_constrain[i] = v[i + 1] - (v[i] + a[i] * dt)
    all_constrain = vertcat(x_constrain, y_constrain, theta_constrain, v_constrain)
    ub_constrains_g = np.zeros([4 * (N - 1)])
    lb_constrains_g = np.zeros([4 * (N - 1)])

    # define cost function f
    cost = 0
    for i in range(N):
        # deviation
        cost += 20/N**3 * (N-i)**4 * (x[i] - route[i][0]) ** 2
        cost += 20/N**3 * (N-i)**4 * (y[i] - route[i][1]) ** 2
        # control cost
        if i < N-2:
            cost += 5 * N * steer[i] ** 2
            cost += 0.01 * N * a[i] ** 2

            cost += 20 * N * (steer[i+1] - steer[i]) ** 2
            # cost += 0.1 * N * (a[i + 1] - a[i]) ** 2

    nlp = {'x': all_vars,
           'f': cost,
           'g': all_constrain}
    S = nlpsol('S', 'ipopt', nlp, {"print_time": False,
                                   "ipopt": {"print_level": 0}})
    result = S(x0=initial_value, lbg=lb_constrains_g, ubg=ub_constrains_g,
               lbx=lb_constrains_vars, ubx=ub_constrains_vars)

    def print_result():
        print('route_x', [i[0] for i in route])
        print('x', result['x'][0:N])
        print('route_y', [i[1] for i in route])
        print('y', result['x'][N:2*N])

        print('theta', result['x'][2*N:3*N])
        print('v', result['x'][3*N:4*N])
        print('vx', result['x'][3*N:4*N] * np.cos(result['x'][2*N:3*N]))
        print('vy', result['x'][3*N:4*N] * np.sin(result['x'][2*N:3*N]))

        print('steer', result['x'][4*N:5*N-1])
        print('a', result['x'][5*N-1:6*N-2])

    # print_result()
    steer = float(result['x'][4*N+1])
    a = float(result['x'][5*N])
    x = result['x'][0:N].elements()
    y = result['x'][N:2*N].elements()

    return a, steer, x, y


def short_term_MPC(ego_car, route, dt, N):
    # initial state
    x_init = [ego_car.x] * N
    y_init = [ego_car.y] * N
    theta_init = [ego_car.theta] * N
    v_init = [ego_car.v] * N
    steer_init = [ego_car.steer] * (N - 1)
    a_init = [MAX_a] * (N - 1)
    initial_value = x_init + y_init + theta_init + v_init +\
                    steer_init + a_init

    # vars to be optimize
    x = SX.sym('x', N)
    y = SX.sym('y', N)
    theta = SX.sym('theta', N)
    v = SX.sym('v', N)
    steer = SX.sym('steer', N - 1)
    a = SX.sym('a', N - 1)
    all_vars = vertcat(x, y, theta, v, steer, a)

    # vars upper bound
    ub_constrains_x = np.array([ego_car.x]+[INF] * (N-1))
    ub_constrains_y = np.array([ego_car.y]+[INF] * (N-1))
    ub_constrains_theta = np.array([ego_car.theta]+[INF] * (N-1))
    ub_constrains_v = np.array([ego_car.v]+[MAX_v] * (N-1))
    ub_constrains_steer = np.array([ego_car.steer]+[MAX_steer] * (N-2))
    ub_constrains_a = np.array([ego_car.a]+[MAX_a] * (N-2))
    ub_constrains_vars = np.hstack([ub_constrains_x, ub_constrains_y,
                                    ub_constrains_theta, ub_constrains_v,
                                    ub_constrains_steer, ub_constrains_a])
    # vars lower bound
    lb_constrains_x = np.array([ego_car.x]+[-INF] * (N-1))
    lb_constrains_y = np.array([ego_car.y]+[-INF] * (N-1))
    lb_constrains_theta = np.array([ego_car.theta]+[-INF] * (N-1))
    lb_constrains_v = np.array([ego_car.v]+[-MIN_v] * (N-1))
    lb_constrains_steer = np.array([ego_car.steer]+[MIN_steer] * (N-2))
    lb_constrains_a = np.array([ego_car.a]+[MIN_a] * (N-2))
    lb_constrains_vars = np.hstack([lb_constrains_x, lb_constrains_y,
                                    lb_constrains_theta, lb_constrains_v,
                                    lb_constrains_steer, lb_constrains_a])

    # define constrain function g (variables update equation)
    x_constrain = SX.sym('x_constrain', N - 1)
    y_constrain = SX.sym('y_constrain', N - 1)
    theta_constrain = SX.sym('theta_constrain', N - 1)
    v_constrain = SX.sym('v_constrain', N - 1)

    SCALE = 0.002
    for i in range(N-1):
        theta_diff = atan(tan(steer[i]) / 2) * v[i] * dt * SCALE
        # theta_diff = steer[i] * dt

        x_constrain[i] = x[i + 1] - (x[i] + v[i] * dt * np.cos(theta[i]))
        y_constrain[i] = y[i + 1] - (y[i] + v[i] * dt * np.sin(theta[i]))
        theta_constrain[i] = theta[i + 1] - (theta[i] - theta_diff)
        v_constrain[i] = v[i + 1] - (v[i] + a[i] * dt)
    all_constrain = vertcat(x_constrain, y_constrain, theta_constrain, v_constrain)
    ub_constrains_g = np.zeros([4 * (N - 1)])
    lb_constrains_g = np.zeros([4 * (N - 1)])

    # define cost function f
    cost = 0
    for i in range(N):
        # deviation
        cost += (x[i] - route[i][0]) ** 2
        cost += (y[i] - route[i][1]) ** 2

    nlp = {'x': all_vars,
           'f': cost,
           'g': all_constrain}
    S = nlpsol('S', 'ipopt', nlp, {"print_time": False,
                                   "ipopt": {"print_level": 0}})
    result = S(x0=initial_value, lbg=lb_constrains_g, ubg=ub_constrains_g,
               lbx=lb_constrains_vars, ubx=ub_constrains_vars)

    def print_result():
        print('route_x', [i[0] for i in route])
        print('x', result['x'][0:N])
        print('route_y', [i[1] for i in route])
        print('y', result['x'][N:2*N])

        print('theta', result['x'][2*N:3*N])
        print('v', result['x'][3*N:4*N])
        print('vx', result['x'][3*N:4*N] * np.cos(result['x'][2*N:3*N]))
        print('vy', result['x'][3*N:4*N] * np.sin(result['x'][2*N:3*N]))

        print('steer', result['x'][4*N:5*N-1])
        print('a', result['x'][5*N-1:6*N-2])

    # print_result()
    steer = float(result['x'][4*N+1])
    a = float(result['x'][5*N])
    x = result['x'][0:N].elements()
    y = result['x'][N:2*N].elements()
    return a, steer, x, y


def build_long_term_larget(track, ind, pos, dt, N):
    desired_v = 80
    dist_travel = desired_v * dt

    def get_point(start, end, d_to_go):
        x0, y0 = start
        x1, y1 = end
        dy = y1-y0
        dx = x1-x0
        d = np.linalg.norm((dx, dy))

        x = x0+d_to_go*dx/d
        y = y0+d_to_go*dy/d

        return np.array((x, y))

    cur_pos = np.array(pos)
    ind = ind % len(track)
    cur_target = np.array(track[ind][2:4])

    result = [pos]
    for i in range(N-1):
        remain_dist = np.linalg.norm(cur_target - cur_pos) - dist_travel
        if remain_dist > 0:
            p = get_point(cur_pos, cur_target, dist_travel)
            result.append(p)
            cur_pos = p
        else:
            # must ensure distance between 2 target points larger than dist_travel
            cur_pos = cur_target
            ind = (ind + 1) % len(track)
            cur_target = np.array(track[ind][2:4])

            p = get_point(cur_pos, cur_target, -remain_dist)
            result.append(p)
            cur_pos = p
    return result


def main():
    env = gym.make('CarRacing-v0')
    done = False

    env.reset()
    car = env.unwrapped.car
    w = car.wheels[0]
    dt = 1/FPS
    prev_a = MAX_a
    prev_steer = 0

    total_reward = 0
    for i in range(1000000):
        print('########################')

        ind = env.unwrapped.tile_visited_count
        ind = ind % len(env.unwrapped.track)
        target = env.unwrapped.track[ind][2:4]

        x, y = car.hull.position
        vx, vy = w.linearVelocity * 1
        theta = atan2(vy, vx)
        dtheta = 0
        obs = x, y, theta, vy, vx, dtheta, target

        long_term_N = 30
        long_term_target = build_long_term_larget(env.unwrapped.track, ind, (x, y), dt, long_term_N)

        ego_car = Car(obs, (prev_a, prev_steer))
        t = time()
        a, steer, x, y = long_term_MPC(ego_car, long_term_target, dt, long_term_N)
        print('long term solve time:', time()-t)

        short_term_N = 5
        short_term_target = list(zip(x, y))[:short_term_N]
        t = time()
        a, steer, x, y = short_term_MPC(ego_car, short_term_target, dt, short_term_N)
        print('short term solve time:', time() - t)

        prev_a, prev_steer = a, steer
        print(a, steer, np.linalg.norm((vx, vy)))

        a = a / MAX_a
        steer = steer
        if a > 0:
            action = steer, a / 10, 0
        else:
            action = steer, 0, -a
        _, r, done, _ = env.step(action)
        total_reward += r

        env.render()

    print(total_reward)


if __name__ == '__main__':
    main()

