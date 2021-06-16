from random import random, seed
from math import sqrt, cos, sin, pi

seed("abc")


def output_points(pts, attributes, fh):
    fh.write("wkt")
    fh.write(";")
    fh.write(";".join(["attrib" + str(i) for (i, attrib) in enumerate(attributes[0])]))
    fh.write("\n")
    for pt, attribs in zip(pts, attributes):
        attribs = ";".join(map(str, attribs))
        fh.write(f"POINT({pt[0]} {pt[1]});{attribs}\n")


def random_circle_vertices(n=10, cx=0, cy=0):
    """Returns a list with n random vertices in a circle

    Method according to:

    http://www.anderswallin.net/2009/05/uniform-random-points-in-a-circle-using-polar-coordinates/
    """
    #     import fractions
    #     from gmpy2 import mpfr
    #     import gmpy2
    #     gmpy2.get_context().precision = 53 * 4

    vertices = []
    for _ in range(n):
        r = sqrt(random())
        t = 2 * pi * random()
        x = r * cos(t)
        y = r * sin(t)
        vertices.append((x + cx, y + cy))
    vertices = list(set(vertices))
    vertices.sort()
    return vertices
