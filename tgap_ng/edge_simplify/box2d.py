X = 0
Y = 1


def increase_box(box, pt):
    """Increase the region covered by box to include point pt
    
    Modifier function.
    """
    box[0][X] = min(box[0][X], pt[X])
    box[0][Y] = min(box[0][Y], pt[Y])

    box[1][X] = max(box[1][X], pt[X])
    box[1][Y] = max(box[1][Y], pt[Y])


def init_box(points):
    """Obtain a tight fitting axis-aligned box around point set"""
    it = iter(points)
    # init box with first point
    pt = next(it)
    min_ = [pt[X], pt[Y]]
    max_ = [pt[X], pt[Y]]
    box = (min_, max_)
    # add next points to it
    for pt in it:
        increase_box(box, pt)
    return box


def brute_box(points):
    """Obtain a tight fitting axis-aligned box around point set"""
    xmin = min(points, key=lambda x: x[0])[0]
    ymin = min(points, key=lambda x: x[1])[1]
    xmax = max(points, key=lambda x: x[0])[0]
    ymax = max(points, key=lambda x: x[1])[1]
    return ([xmin, ymin], [xmax, ymax])


def as_polygon(box2d):
    (xmin, ymin), (xmax, ymax) = box2d
    return f"POLYGON(({xmin} {ymin}, {xmin} {ymax}, {xmax} {ymax}, {xmax} {ymin}, {xmin} {ymin}))"


if __name__ == "__main__":
    from .utils import random_circle_vertices

    pts = random_circle_vertices(1000)
    assert init_box(pts) == brute_box(pts)
