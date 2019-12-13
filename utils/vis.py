import pickle

import matplotlib.pyplot as plt


def plot_waypoints(long_term_pnts, short_term_pnts):

    def _to_lons_lats(pnts):
        return pnts[:len(pnts) // 2], pnts[len(pnts) // 2 : len(pnts)]
    long_term_lons, long_term_lats = _to_lons_lats(long_term_pnts)
    short_term_lons, short_term_lats = _to_lons_lats(short_term_pnts)
    plt.scatter(long_term_lats, long_term_lons, c='k', s=2)
    plt.scatter(short_term_lats, short_term_lons, c='r', s=2)
    plt.axis('equal')
    plt.show()


def draw_running_result(fname, track, trajectory):
    plt.figure()
    plt.plot(track['x'], track['y'], linewidth=30, color='gray')
    plt.plot(trajectory['x'], trajectory['y'], color='red')
    plt.savefig(fname, dpi=500)
    # plt.show()


if __name__=='__main__':

    mode = "mixed"
    # with open("../saved/episode_mpc.pkl", "rb") as f:
    # with open("../saved/episode_mixed.pkl", "rb") as f:
    with open("../saved/episode_{}.pkl".format(mode), "rb") as f:
        dic = pickle.load(f)
        track, trajectory = dic['track'], dic['traj']

    draw_running_result("{}.png".format(mode), track, trajectory)