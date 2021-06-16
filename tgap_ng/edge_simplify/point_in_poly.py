# FIXME: check does polygon needs to be closed (i.e. pt[0] == pt[-1]?

# does the following give same result for all methods?
# [Point(x=186394.49738688886, y=311836.78388576297, srid=28992),
# Point(x=186396.512, y=311855.23, srid=28992),
# Point(x=186397.404, y=311869.598, srid=28992)]
# (186397.101, 311864.729)


def point_in_polygon__pnpoly(polygon, pt):
    """ fast """
    # by W. Randolph Franklin (WRF)  https://wrf.ecse.rpi.edu//Research/Short_Notes/pnpoly.html
    # also check https://stackoverflow.com/questions/36399381/whats-the-fastest-way-of-checking-if-a-point-is-inside-a-polygon-in-python

    if polygon[0] != polygon[-1]:
        polygon = polygon[:]
        polygon.append(polygon[0])

    inside = False
    j = len(polygon) - 1
    for i in range(len(polygon)):
        if (polygon[i][1] > pt[1]) != (polygon[j][1] > pt[1]) and (
            pt[0]
            < (
                (polygon[j][0] - polygon[i][0])
                * (pt[1] - polygon[i][1])
                / (polygon[j][1] - polygon[i][1])
                + polygon[i][0]
            )
        ):
            inside = not inside
        j = i
    return inside


def point_in_polygon__ray_tracing(polygon, pt):
    """ slowest """
    poly = polygon

    if polygon[0] != polygon[-1]:
        polygon = polygon[:]
        polygon.append(polygon[0])

    x, y = pt
    n = len(poly)
    inside = False
    p2x = 0.0
    p2y = 0.0
    xints = 0.0
    p1x, p1y = poly[0]
    for i in range(n + 1):
        p2x, p2y = poly[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xints = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xints:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside


def point_in_polygon__is_inside_sm(polygon, point):
    """ fastest """
    # https://github.com/sasamil/PointInPolygon_Py/blob/master/pointInside.py#L134
    if polygon[0] != polygon[-1]:
        polygon = polygon[:]
        polygon.append(polygon[0])

    length = len(polygon) - 1
    dy2 = point[1] - polygon[0][1]
    intersections = 0
    ii = 0
    jj = 1
    while ii < length:
        dy = dy2
        dy2 = point[1] - polygon[jj][1]
        # consider only lines which are not completely above/below/right from the point
        if dy * dy2 <= 0.0 and (
            point[0] >= polygon[ii][0] or point[0] >= polygon[jj][0]
        ):
            # non-horizontal line
            if dy < 0 or dy2 < 0:
                F = dy * (polygon[jj][0] - polygon[ii][0]) / (dy - dy2) + polygon[ii][0]
                if (
                    point[0] > F
                ):  # if line is left from the point - the ray moving towards left, will intersect it
                    intersections += 1
                elif point[0] == F:  # point on line
                    return bool(2)
            # point on upper peak (dy2=dx2=0) or horizontal line (dy=dy2=0 and dx*dx2<=0)
            elif dy2 == 0 and (
                point[0] == polygon[jj][0]
                or (
                    dy == 0
                    and (point[0] - polygon[ii][0]) * (point[0] - polygon[jj][0]) <= 0
                )
            ):
                return bool(2)
        ii = jj
        jj += 1
    # print 'intersections =', intersections
    return bool(intersections & 1)


def test():
    ring = [(185602.937, 314563.19), (185607.755, 314580.246), (185614.072, 314601.833)]

    pt = (185613.498, 314564.17)
    ring.append(ring[0])
    results = [
        point_in_polygon__pnpoly(ring, pt),
        point_in_polygon__ray_tracing(ring, pt),
        point_in_polygon__is_inside_sm(ring, pt),
    ]
    print(results)


def test2():
    from .utils import random_circle_vertices, output_points

    pts = random_circle_vertices(
        # 1_000_000
        10_000
    )

    size = 0.5

    # square
    # poly = [(-size, -size), (-size, +size), (+size, +size), (+size, -size), (-size, -size)]

    # poly = [(-size, -size), (0, +size), (+size, -size)]#, (-size, -size)]

    # invalid polygonal ring: self intersecting (bow tie)
    poly = [
        (-size, -size),
        (+size, +size),
        (-size, +size),
        (+size, -size),
        (-size, -size),
    ]

    attributes = []
    for pt in pts:
        attribs = [
            point_in_polygon__pnpoly(poly, pt),
            point_in_polygon__ray_tracing(poly, pt),
            point_in_polygon__is_inside_sm(poly, pt),
        ]
        attributes.append(attribs)

    with open("/tmp/pts.wkt", "w") as fh:
        output_points(pts, attributes, fh)


if __name__ == "__main__":
    test()
