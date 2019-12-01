from math import cos, sin


def to_absolute_coords(car_x, car_y, theta, xs, ys, combine=True):
    lons, lats = [], []

    def _to_absolute_coord(x, y):
        x_ = cos(-theta) * x + sin(-theta) * y
        y_ = - sin(-theta) * x + cos(-theta) * y

        lons.append(x_ + car_x)
        lats.append(y_ + car_y)

    for x, y in zip(xs, ys):
        _to_absolute_coord(x, y)

    if not combine:
        return lons, lats
    return lons + lats


def to_relative_coords(car_x, car_y, theta, xs, ys, combine=True):
    lons, lats = [], []

    def _to_relative_coord(x, y):
        x -= car_x
        y -= car_y

        lons.append(cos(theta) * x + sin(theta) * y)
        lats.append(- sin(theta) * x + cos(theta) * y)

    for x, y in zip(xs, ys):
        _to_relative_coord(x, y)

    if not combine:
        return lons, lats
    return lons + lats
