from simplegeom.geometry import LineString, LinearRing, Polygon

from .vectorops import sub, cross, norm
from .box2d import init_box, increase_box, as_polygon
from .point_in_poly import point_in_polygon__is_inside_sm as is_point_in_polygon

# from .point_in_poly import point_in_polygon__pnpoly as is_point_in_polygon

import unittest
import math

# Point-Line Distance : the signed distance between the line AB and the point C:
# fn pointLineDist2 pA pB pC = (
# 	local vAB=pB-pA
# 	local vAC=pC-pA
# 	(cross vAB vAC)/(length vAB)

X = 0
Y = 1


def point_line_distance(line, pt):
    """The positive distance from the point to the line"""
    pA, pB = line
    pC = pt
    vAB = sub(pB, pA)
    vAC = sub(pC, pA)
    num = cross(vAB, vAC)
    den = norm(vAB)
    dist = abs(num) / den
    return dist


class TestDistanceCalculation(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_simple(self):
        self.assertEqual(point_line_distance([(0, 0), (10, 0)], (5, 5)), 5.0)
        self.assertEqual(point_line_distance([(0, 0), (10, 0)], (5, -5)), 5.0)
        self.assertEqual(point_line_distance([(0, -5), (0, 5)], (-5, 0)), 5.0)
        self.assertEqual(point_line_distance([(0, -5), (0, 5)], (5, 0)), 5.0)

    def test_approximate(self):
        self.assertAlmostEqual(
            point_line_distance([(-10, -10), (10, 10)], (-1, 1)), math.sqrt(2.0)
        )
        self.assertAlmostEqual(
            point_line_distance([(-10, -10), (10, 10)], (1, -1)), math.sqrt(2.0)
        )
        self.assertAlmostEqual(
            point_line_distance([(10, -10), (-10, 10)], (1, 1)), math.sqrt(2.0)
        )
        self.assertAlmostEqual(
            point_line_distance([(10, -10), (-10, 10)], (-1, -1)), math.sqrt(2.0)
        )


def simplify_reumann_witkam(polyline, pp, tolerance, DEBUG):

    #    own_points = set([(pt[0], pt[1]) for pt in polyline])

    p0 = 0
    p1 = 1

    #    box = polyline.envelope # init_box(ring)
    #    ring = [polyline[p0], polyline[p1]]
    #    for i in range(2, len(polyline)):
    #        increase_box(box, polyline[i])
    #    others = set(pp.quadtree.range_search([(box.xmin, box.ymin), (box.xmax, box.ymax)]))

    keep = [True] * len(polyline)

    is_loop = polyline[0] == polyline[-1]
    remaining_ct = len(polyline)

    # the first points defines the line to check the other points against
    p0 = 0
    p1 = 1
    pi = pj = p1

    # keep first point
    keep[p0] = True

    ring = [polyline[p0], polyline[p1]]
    box = init_box(ring)

    measure = float("+inf")
    for i in range(2, len(polyline)):
        pi = pj
        pj += 1  # advance the index for the point to be checked

        if DEBUG:
            print("Debug info for RW-simplify")
            print(f"p0 := {p0}")
            print(f"p1 := {p1}")
            print(f" pi := {pi}")
            print(f" pj := {pj}")
            print("")
        ring.append(polyline[pj])
        increase_box(box, polyline[pj])
        dist = point_line_distance([polyline[p0], polyline[p1]], polyline[pj])

        # the points of the line that we have kept so far / still need to check? (self intersect)
        own_points = set([(pt[0], pt[1]) for pt in ring])
        # FIXME:
        # if we are going to keep this point (and shift the initial direction line/box),
        # we do not need the whole topology preserving circus
        # only, if the point is potentially to be removed, we need to check for violations
        #
        # at this moment this does not have the correct order of operations!

        if DEBUG:

            with open("/tmp/init_line.wkt", "w") as fh:
                lnstr = "LINESTRING({0[0][0]} {0[0][1]}, {0[1][0]} {0[1][1]})".format(
                    [polyline[p0], polyline[p1]]
                )
                fh.write("wkt\n")
                fh.write(lnstr)
                fh.write("\n")

            with open("/tmp/point__check.wkt", "w") as fh:
                lnstr = "POINT({0[0]} {0[1]})".format(polyline[pj])
                fh.write("wkt\n")
                fh.write(lnstr)
                fh.write("\n")

            simplified = []
            for preserve, pt in zip(keep, polyline):
                if preserve:
                    simplified.append(pt)

            with open("/tmp/simplified_line.wkt", "w") as fh:
                fh.write("wkt\n")
                fh.write("{}".format(LineString(simplified, polyline.srid)))
                fh.write("\n")

            if len(ring) >= 3:
                with open("/tmp/keepfree_poly.wkt", "w") as fh:
                    tmp = ring[:]
                    tmp.append(ring[0])
                    fh.write("wkt\n")
                    fh.write(
                        "{}".format(
                            Polygon(
                                shell=LinearRing(tmp, polyline.srid), srid=polyline.srid
                            )
                        )
                    )
                    fh.write("\n")

        # FIXME: next to the distance, we also should check whether the
        # polygon formed by the points from p0...pj is still empty of other
        # points
        # -- polygon should be formed here / to linear ring a point should
        #    be added, and around this polygon we should query the quadtree
        #    for points in the neighbourhood
        #    then we should do a point in polygon test for points not being
        #    part of the polygon, best if such a point in polygon test
        #    is robust against polygons not being 'valid' (like bow ties)
        if pp is not None:
            # own = set((pt.x, pt.y) for pt in ring)

            # FIXME: deal with points of own line that we will also find
            # with the quadtree

            # FIXME: we could also do this search at the beginning
            # at the expense of doing some more iterations here
            # and also having to keep this set then updated

            others = pp.quadtree.range_search(box)

            if DEBUG:
                with open("/tmp/qt_box.wkt", "w") as fh:
                    fh.write("wkt\n")
                    fh.write(as_polygon(box))
                    fh.write("\n")
                with open("/tmp/close_points.wkt", "w") as fh:
                    fh.write("wkt\n")
                    for pt in others:
                        fh.write(f"POINT({pt[0]} {pt[1]})")
                        fh.write("\n")

            topology_problem = False
            for other in others:
                # FIXME: this is not okay, as self intersection potentially can happen
                if DEBUG:
                    print(f"checking {other}")
                    print(ring)
                    print(other)
                if other in own_points:
                    if DEBUG:
                        print("- other part of line being simplified")
                    # FIXME: potential self-intersection
                    # - to fix, make set of ring object and use that
                    continue

                if is_point_in_polygon(ring, other):
                    if DEBUG:
                        print("- other lies in sweep-poly (blocking!)")
                    topology_problem = True
                    break  # ?
                else:
                    if DEBUG:
                        print("- other not in sweep-poly")

            if DEBUG:
                input(f"line written as debugging requested (DEBUG={DEBUG})")

            if topology_problem:
                # we keep the point at index pi
                keep[pi] = True
                # make a new line for checking the remaining points with
                p0 = pi
                p1 = pj
                ring = [polyline[p0], polyline[p1]]
                box = init_box(ring)
                #                if dist < measure:
                #                    measure = dist
                # continue main loop, check remaining points
                continue

        if is_loop and remaining_ct <= 4:
            # we need 4 points in a loop edge, no further simplification
            # set measure to +inf, preventing from picking this edge again for
            # simplification
            measure = float("inf")
            break

        # FIXME: temporary stop criterion, always keep 3 points in any edge
        # rationale: areas keep some size with 3 (non-collinear) points
        #            although no guarantees on collinearity, reduces chance
        #            on face collapsing into line (degeneracy)
        if remaining_ct <= 3:
            measure = float("inf")
            break

        if dist <= tolerance:
            # do not keep this point
            # FIXME: which point to keep pi or pj?
            if pp is not None:
                pp.quadtree.remove((polyline[pi][X], polyline[pi][Y]))
            #                others.discard((polyline[pi][X], polyline[pi][Y]))
            remaining_ct -= 1
            keep[pi] = False
            continue

        else:
            # we want to keep the point at index pi
            keep[pi] = True
            # make a new line for checking the remaining points with
            p0 = pi
            p1 = pj
            ring = [polyline[p0], polyline[p1]]
            box = init_box(ring)
            if dist < measure:
                measure = dist

    # keep last point
    keep[-1] = True

    # first and last point should be kept
    assert keep[0] == True
    assert keep[len(polyline) - 1] == True

    # get the simplified polyline
    simplified = []
    for preserve, pt in zip(keep, polyline):
        if preserve:
            simplified.append(pt)

    return (
        LineString(simplified, polyline.srid),
        measure,
    )  # FIXME: check if new tolerance calculation is correct


def main():
    """Main - invoke tests"""
    # unittest.main()

    polyline = [
        (0, 2),
        (3, 5),
        (5.5, 4.5),
        (6, 5),
        (8, 4),
        (10, 0),
        (12, 1),
        (13, 0),
        (14, 5),
    ]
    simplify_reumann_witkam(polyline, float(10))


def test():
    tolerance = 2.6458333333333335
    line = [
        (189860.459, 314416.086),
        (189857.269, 314414.272),
        (189855.988, 314416.525),
        (189859.178, 314418.34),
        (189860.459, 314416.086),
    ]
    simplify_reumann_witkam(line, None, tolerance, False)


if __name__ == "__main__":
    test()
    # main()test
