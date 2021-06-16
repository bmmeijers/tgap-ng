def orient2d(pa, pb, pc):
    """Twice signed area of triangle formed by points a, b and c
    
    Direction from pa to pc, via pb, where returned value is as follows:
    
    left : + [ = ccw ]
    straight : 0.
    right : - [ = cw ]
    
    :param pa: point
    :type pa: sequence
    :param pb: point
    :type pb: sequence
    :param pc: point
    :type pc: sequence
    :returns: double
    """
    acx = pa[0] - pc[0]
    bcx = pb[0] - pc[0]
    acy = pa[1] - pc[1]
    bcy = pb[1] - pc[1]
    return acx * bcy - acy * bcx


def sign(value):
    if value > 0:
        return 1
    elif value < 0:
        return -1
    else:
        return 0


# def on_triangle(pt, a, b, c):
#    """
#    Using signed distances to edges (halfplanes)

#    Returns true when ``pt'' is on triangle (formed by point ``a'', ``b''
#    and ``c''), which means: not on exterior (returns False); On interior or
#    on boundary (returns True)
#    """
#    dists = orient2d(a, b, pt), orient2d(b, c, pt), orient2d(c, a, pt)
#    # find a non-zero distance
#    for d in dists:
#        if d != 0:
#            s = sign(d)
#            break
#    else:
#        # all distances zero, so point tested is on the 3 edges
#        # of the triangle, assume it is inside
#        # (although if degenerate triangle - where triangle is a line segment -
#        #  the point can be outside as well)
#        return True
#    # here we have a non-zero distance
#    # compare whether other non-zero distances have same sign
#    for d in dists:
#        if d == 0:
#            continue
#        elif sign(d) != s:
#            # if not, then we are outside
#            return False
#        else:
#            continue
#    # otherwise, we are inside
#    return True


# def on_triangle(p, p0, p1, p2):
#    """
#    Using bary-centric logic (twice as fast as halfplane based method)

#    Returns true when ``pt'' is on triangle (formed by point ``a'', ``b''
#    and ``c''), which means: not on exterior (returns False); On interior or
#    on boundary (returns True)

#    From: https://stackoverflow.com/a/34093754
#    """
#    dX = p[0] - p2[0]
#    dY = p[1] - p2[1]
#    dX21 = p2[0] - p1[0]
#    dY12 = p1[1] - p2[1]
#    D = dY12 * (p0[0] - p2[0]) + dX21 * (p0[1] - p2[1])
#    s = dY12 * dX + dX21 * dY
#    t = (p2[1] - p0[1]) * dX + (p0[0] - p2[0]) * dY
#    if D < 0:
#        return s <= 0 and t <= 0 and s+t >= D
#    else:
#        return s >= 0 and t >= 0 and s+t <= D


def on_triangle(p_test, p0, p1, p2):
    """
    Using bary-centric logic (twice as fast as halfplane based method)

    Returns true when ``pt'' is on triangle (formed by point ``a'', ``b'' 
    and ``c''), which means: not on exterior (returns False); On interior or 
    on boundary (returns True)

    From: https://stackoverflow.com/a/34093754
    """
    dX = p_test[0] - p0[0]
    dY = p_test[1] - p0[1]
    dX20 = p2[0] - p0[0]
    dY20 = p2[1] - p0[1]
    dX10 = p1[0] - p0[0]
    dY10 = p1[1] - p0[1]

    s_p = (dY20 * dX) - (dX20 * dY)
    t_p = (dX10 * dY) - (dY10 * dX)
    D = (dX10 * dY20) - (dY10 * dX20)

    if D > 0:
        return (s_p >= 0) and (t_p >= 0) and (s_p + t_p) <= D
    else:
        return (s_p <= 0) and (t_p <= 0) and (s_p + t_p) >= D


def test_degenerate():
    assert not on_triangle([11, 45], [45, 45], [45, 45], [44, 45])


def _test():
    """unit test for on_triangle"""
    a, b, c = (10, 10), (25, 10), (25, 25)
    # 6 permutations to form the triangle
    T = [(a, b, c), (a, c, b), (b, a, c), (b, c, a), (c, a, b), (c, b, a)]
    inside = set(
        [
            (10, 10),
            (15, 10),
            (20, 10),
            (25, 10),
            (15, 15),
            (20, 15),
            (25, 15),
            (20, 20),
            (25, 20),
            (25, 25),
        ]
    )

    for _ in range(10000):
        for t in range(6):
            p0, p1, p2 = T[t]
            for i in range(-10, 50, 5):
                for j in range(0, 31, 5):
                    pt = (i, j)
                    on = on_triangle(pt, p0, p1, p2)
                    if pt in inside:
                        assert on, "p<{0}>, t<{1}, {2}, {3}>".format(pt, p0, p1, p2)
                    else:
                        assert not on, "p<{0}>, t<{1}, {2}, {3}>".format(pt, p0, p1, p2)


if __name__ == "__main__":
    #    _test()
    test_degenerate()
