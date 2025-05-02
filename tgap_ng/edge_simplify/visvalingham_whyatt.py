from .. import pqdict

from math import fabs
from operator import itemgetter
from simplegeom.geometry import LineString

from .ontriangle import on_triangle, orient2d


def area(p0, p1, p2):
    """Calculate the area of the triangle formed by three points"""
    det = orient2d(p0, p1, p2)
    area = fabs(0.5 * det)
    return area


def dist(pa, pb):
    dx = pb[0] - pa[0]
    dy = pb[1] - pa[1]
    return (dx ** 2 + dy ** 2) ** 0.5


def output_points(pts, fh):
    for pt in pts:
        fh.write("POINT({0[0]} {0[1]})\n".format(pt))


#def simplify(line, pp, tolerance=float("inf"), DEBUG=False):
def simplify(pp, edge_id, tolerance=float("inf"), DEBUG=False):
    """Simplifies a polyline with Visvalingham Whyatt algorithm,
    i.e. bottom up
    """
    # FIXME: Do we need an additional step to split a polyline
    # in 2 pieces when it is a loop? Will lead to different simplification at
    # least
    #
    line = pp.edges[edge_id].geometry

    start_node_id = pp.edges[edge_id].start_node_id
    end_node_id = pp.edges[edge_id].end_node_id

    #print(pp.nodes[start_node_id].star.index(edge_id))
    star_start = pp.nodes[start_node_id].star
    star_start_idx = star_start.index(edge_id)
    # neighbouring edges at the start node
    signed_neighbour_ids = [star_start[(star_start_idx + 1) % len(star_start)], star_start[(star_start_idx - 1) % len(star_start)]]
    edge_ids_at_start = set([ngb_edge_id if ngb_edge_id > 0 else ~ngb_edge_id for ngb_edge_id in signed_neighbour_ids])

    #print(pp.nodes[end_node_id].star.index(~edge_id))
    star_end = pp.nodes[end_node_id].star
    star_end_idx = star_end.index(~edge_id)
    # neighbouring edges at the end node
    signed_neighbour_ids = [star_end[(star_end_idx + 1) % len(star_end)], star_end[(star_end_idx - 1) % len(star_end)]]
    edge_ids_at_end = set([ngb_edge_id if ngb_edge_id > 0 else ~ngb_edge_id for ngb_edge_id in signed_neighbour_ids])

    has_short_cut_already = False
    short_cut_edge_ids = edge_ids_at_start.intersection(edge_ids_at_end)
    for short_cut_id in short_cut_edge_ids:
        if len(pp.edges[short_cut_id].geometry) == 2:
            has_short_cut_already = True
            break

    prv = [None] * len(line)
    nxt = [None] * len(line)
    for i in range(len(line)):
        prv[i] = i - 1
        nxt[i] = i + 1

    # first does not have prv and
    # last does not have nxt
    prv[0] = None
    nxt[-1] = None

    if DEBUG:
        print(line)
        print(prv)
        print(nxt)

    # measures for points
    # first and last do not get any measure
    oseq = pqdict.pqdict()
    for i in range(1, len(line) - 1):

        size = area(line[prv[i]], line[i], line[nxt[i]])
        base = dist(line[prv[i]], line[nxt[i]])

        if DEBUG:
            print(i)
            print(prv[i])
            print(nxt[i])
            print(base)

        if base == 0:
            eps = float("inf")
        else:
            eps = size / (0.5 * base)
        oseq[i] = eps

    if DEBUG:
        for _k in oseq:
            print(_k, "->", oseq[_k])

    # FIXME:
    # loops can also be formed by 2 edges,
    # that can end up 'on top' of each other
    is_loop = line[0] == line[-1]

    remaining_ct = len(line)

    measure = float("inf")

    has_blocker = False
    while oseq:
        idx, measure, = oseq.popitem()
        
        #        print('try removing {}'.format(line[idx]))
        if measure > tolerance:
            has_blocker = False
            # if has_blockers:
            #     raise ValueError(f'{idx} {has_blockers}')
            break
        if is_loop and remaining_ct <= 4:
            # we need 4 points in a loop edge, no further simplification
            # set measure to +inf, preventing from picking this edge again for
            # simplification
            #print(f'no simplify further for {edge_id} (loop)')
            measure = float("inf")
            break

        # FIXME: temporary stop criterion, always keep 3 points in any edge
        # rationale: areas keep some size with 3 (non-collinear) points
        #            although no guarantees on collinearity, reduces chance
        #            on face collapsing into line (degeneracy)
        if has_short_cut_already and remaining_ct <= 3:
            #print(f'no simplify further for {edge_id} (3 points kept, as other short cut of 2 points already exists)')
            measure = float("inf")
            break

        # -- check if the triangle is empty
        ###        print prv[idx], idx, nxt[idx]
        start, mid, end = line[prv[idx]], line[idx], line[nxt[idx]]

        if DEBUG:  # True:
            with open("/tmp/triangle_pts.wkt", "w") as fh:
                fh.write("wkt\n")
                output_points([start, mid, end], fh)

        # rectangle that lies around this triangle
        xmin = min(start[0], min(mid[0], end[0]))
        xmax = max(start[0], max(mid[0], end[0]))

        ymin = min(start[1], min(mid[1], end[1]))
        ymax = max(start[1], max(mid[1], end[1]))

        rect = [(xmin, ymin), (xmax, ymax)]

        if DEBUG:  # True:
            with open("/tmp/rect.wkt", "w") as fh:
                fh.write("wkt\n")
                fh.write(
                    """POLYGON(({xmin} {ymin}, {xmax} {ymin}, {xmax} {ymax}, {xmin} {ymax}, {xmin} {ymin}))\n""".format(
                        xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax
                    )
                )

        # by means of the quadtree we find overlapping points
        # the number of overlapping points can already tell us something:
        #
        # if we find exactly 3, this triangle is empty, no need to check
        # the points
        # - however, it depends on the topological config with other edges
        #   whether it is safe to remove the vertex -> end points that form
        #   base of triangle, which also is an edge --> problematic to remove
        #   (area collapse of triangle) -- two edges form a loop
        #
        # if we find less than 3, there is a problem
        # - either this is a degenerate line (with multiple vertices on same
        #   location) -> should be solved while reading the data
        # - or there exists a problem/bug with the quadtree/keeping it up to date
        #   -> point not added in the tree, or the intersection method is
        #      not performing ok...
        #
        # if there is more than 3 points, we could potentially introduce
        # intersection between segments, check if these points lie on interior
        # of triangle that will be collapsed if we remove the point, if not
        # then it is safe to remove the vertex from the polyline
        # (postbox principle)

        has_blocker = False
        overlapping_pts = pp.quadtree.range_search(rect)
        #        overlapping_pts = pp.kdtree.range_search(*rect)
        if DEBUG:  #

            def do_debug():
                with open("/tmp/overlapping.wkt", "w") as fh:
                    fh.write("wkt\n")
                    output_points(overlapping_pts, fh)

                with open("/tmp/quadtree_pts.wkt", "w") as fh:
                    fh.write("wkt\n")
                    output_points([pt for pt in pp.quadtree], fh)

            do_debug()

        # blocks = {}
        # blocked_by = {}
        for pt in overlapping_pts:
            if (
                (pt[0] == start[0] and pt[1] == start[1])
                or (pt[0] == mid[0] and pt[1] == mid[1])
                or (pt[0] == end[0] and pt[1] == end[1])
            ):
                continue
            elif on_triangle(pt, start, mid, end):
                has_blocker = True
                # measure = float("inf") ## omitting this line makes for a sincere drop in performance
                # pp.crosslinks.add_blocker(pt, edge_id)

                # if pt in blockers:
                #     blocks[pt].add(edge_id)
                #     blocked_by[edge_id].add(pt)

                # blocked.append(edge_id)
                # print(f'found overlap for {edge_id}')
                break

        if DEBUG:  #
            print(f"has_blocker := {has_blocker} ")
            input("debugging line")

        if has_blocker:
            continue

        # -- really remove vertex from the line
        pp.quadtree.remove((mid[0], mid[1]))
        # unblocked_edge_ids = pp.crosslinks.remove_blocker(pt)
        # if unblocked_edge_ids:
        #     print("*" * 25, unblocked_edge_ids)

        # unblocked = []
        # if mid in blocks:
        #     for edge_id in blocks[mid]:
        #           blocked_by[edge_id].remove(mid)
        #           if len(blocked_by[edge_id]) == 0:
        #                 unblocked.append(edge_id)

        remaining_ct -= 1

        # make sure we only simplify interior points, and not end points
        # get neighbouring points
        prvidx = prv[idx]
        nxtidx = nxt[idx]
        indices = []
        if prvidx != 0:
            indices.append(prvidx)
        if nxtidx != len(line) - 1:
            indices.append(nxtidx)
        #
        #        for idx in indices:
        #            oseq.pop(idx)
        # link up previous and nxt vertex
        nxt[prvidx] = nxtidx
        prv[nxtidx] = prvidx
        # update measures for those left and right
        for i in indices:
            assert 0 < i < len(line), "out of bounds: {0} {1}".format(i, len(line))
            size = area(line[prv[i]], line[i], line[nxt[i]])
            base = dist(line[prv[i]], line[nxt[i]])
            if base == 0:
                eps = float("inf")
            else:
                eps = size / (0.5 * base)
            oseq[i] = eps
    if remaining_ct == 2:
        measure = float("inf")

    simplified = []
    simplified.append(line[0])
    nxtidx = nxt[0]
    assert nxtidx is not None
    while nxtidx is not None:
        simplified.append(line[nxtidx])
        nxtidx = nxt[nxtidx]
    new_ln = LineString(simplified, line.srid)

    if has_blocker:
        # we exited simplify loop the last point in the line is thus blocked by another point
        # if we would put it back in the queue just to simplify, we most likely find it blocked next time again
        # leading to simplifying over-and-over
        # so just leave this blocked vertex in (an alternative would be to keep administration on which vertex blocks which edge, and update with each edge removal)
        # NOTE, if another vertex is present in the line, but with larger eps than currently, we use this measure, and we will simplify
        # line next time based on threshold of this line
        #raise ValueError(f"{has_blockers} {len(has_blockers)+2} {len(new_ln)}")
        #print(f"no simplify further (blocked) for {edge_id} (with {len(new_ln)} points left)")
        measure = float('inf')

    if DEBUG:
        with open("/tmp/result.wkt", "w") as fh:
            fh.write("wkt\n")
            fh.write(str(new_ln))
        input("paused after simplify")

    has_same_number_of_vertices = len(new_ln) == len(line)
    if has_same_number_of_vertices:
        # print("-" *25, f"not simplified {edge_id} {len(new_ln)} {len(line)} {measure}")
        global CT_NOT_SIMPLIFIED
        CT_NOT_SIMPLIFIED += 1

    else:
        # print("-" *25, f"simplified {edge_id} {len(new_ln)} {len(line)} {measure}")
        global CT_SIMPLIFIED
        CT_SIMPLIFIED += 1
    is_simplified = not has_same_number_of_vertices
    return new_ln, is_simplified, measure

CT_SIMPLIFIED = 0
CT_NOT_SIMPLIFIED = 0

def _test():
    simplify([(0, 0), (10, 0)])
    print((simplify([(0, 0), (5, 5), (7, 0), (10, 0)], 2)))


if __name__ == "__main__":
    _test()
