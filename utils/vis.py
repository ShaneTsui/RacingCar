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