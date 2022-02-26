from collections import namedtuple, defaultdict
from functools import partial
from connection import connection
import time
import math
import simplegeom.geometry
import array
import sys
from geompreds import orient2d
from quadtree import QuadTree

TAU = math.pi * 2 # https://tauday.com/


def vertex_check(pp):
    # BEGIN vertex check
    # check that all edges in the planar partition
    # have their vertices in the quadtree
    with open("/tmp/quadtree_nodes.wkt", "w") as fh:
        print("id;wkt;point_count", file=fh)
        for node in pp.quadtree:
            if node.leaf:
                print(f"{id(node)};{node};{len(node.bucket)}", file=fh)

    missing = []
    points = []
    for edge_id in pp.edges:
        edge = pp.edges[edge_id]
        for pt in edge.geometry:
            if (pt.x, pt.y) not in pp.quadtree:
                missing.append(edge_id)
                points.append(pt)
    output_pp_wkt(pp, "step")
    do_output_quadtree(pp)
    if missing:
        print(set(missing))
        for pt in points:
            print(pt)
    input(f"check - missing vertices in quadtree {set(missing)}")
    # END vertex check


def output_topomap_wkt(tm, name=None):
    """Output topo map to well known text file
    """
    nname = "n.wkt"
    ename = "e.wkt"
    fname = "f.wkt"
    if name is not None:
        nname = name + nname
        ename = name + ename
        fname = name + fname
    with open("/tmp/" + nname, "w") as fh:
        print("id;wkt", file=fh)
        for node in tm.nodes.values():
            print(node.id, ";", node.geometry, file=fh)

    with open("/tmp/" + ename, "w") as fh:
        print("id;sn;en;lf;rf;wkt;attrs", file=fh)
        for he in tm.half_edges.values():
            print(
                he.id,
                ";",
                he.start_node.id,
                ";",
                he.end_node.id,
                ";",
                he.left_face.id,
                ";",
                he.right_face.id,
                ";",
                he.geometry,
                ";",
                he.attrs,
                file=fh,
            )

    with open("/tmp/" + fname, "w") as fh:
        print("id;wkt", file=fh)
        for f in tm.faces.values():
            if not f.unbounded:
                for g in f.multigeometry():
                    print(f.id, ";", g, file=fh)


#
# FIXME: can we use Cython / OpenMP for our data structure?
# See: cython.parallel.prange
# especially obtaining the wheels and areas of faces can be carried out easily in parallel
# https://gist.github.com/zed/2051661
# http://archive.euroscipy.org/talk/6857
#

# -- record types
Edge = namedtuple(
    "Edge",
    "id, start_node_id, start_angle, end_node_id, end_angle, left_face_id, right_face_id, geometry, info",
)  # FIXME: feature class !!!
# FIXME: should we make an additional indirection layer where we do store the feature class
# and other info for objects (point, line, area objects??? that do have a relationship with node, edge, face) ????
# this would mean that we then order these objects for their removal...
Face = namedtuple(
    "Face", "id, mbr_geometry, pip_geometry, edges, info"
)  # FIXME: feature_class_id ==> fc_id ??? Replace by generic info dictionary field?
Node = namedtuple("Node", "id, geometry, star")


# -- lambda function to get positive identifier
# positive_id = lambda _: _ if _ > 0 else ~_
def positive_id(oid):
    """Return a positive identifier (for an edge)"""
    return oid if oid > 0 else ~oid


def csv_line(lst):
    return ",".join(['"{}"'.format(x) for x in list(map(str, lst))])


def output_pp_wkt(pp, name=None):
    """ Output planar partition to well known text file
    """
    nname = "n.wkt"
    ename = "e.csv"
    fname = "f.wkt"
    if name is not None:
        nname = name + "_" + nname
        ename = name + "_" + ename
        fname = name + "_" + fname

    with open("/tmp/" + ename, "w") as fh:
        print(csv_line(["id", "sn", "sa", "en", "ea", "lf", "rf", "step_low", "wkt"]), file=fh)
        for edge in pp.edges.values():
            print(
                csv_line(
                    [
                        edge.id,
                        edge.start_node_id,
                        edge.start_angle,
                        edge.end_node_id,
                        edge.end_angle,
                        edge.left_face_id,
                        edge.right_face_id,
                        edge.info['step_low'],
                        edge.geometry,
                    ]
                ),
                file=fh,
            )

    with open("/tmp/" + fname, "w") as fh:
        print(csv_line(["id", "wkt"]), file=fh)
        for face in pp.faces.values():
            # "POINT({0[0]} {0[1]})".format(face.pip_geometry)
            #            print >> fh, ";".join(map(str, (face.id, "")))
            if face.pip_geometry is not None:
                print(
                    csv_line(
                        [face.id, "POINT({0[0]} {0[1]})".format(face.pip_geometry)]
                    ),
                    file=fh,
                )
            
#            else:
#                print(csv_line([face.id, "POINT EMPTY"]), file=fh)


#    with open("/tmp/"+fname, "w") as fh:
#        print >> fh, "id;wkt"
#        for f in tm.faces.itervalues():
#            if not f.unbounded:
#                for g in f.multigeometry():
#                    print >> fh,  f.id,";", g


class PlanarPartition(object):
    """Planar Partition class"""

    def __init__(self, unbounded_id):
        # FIXME: srid?
        self.unbounded_id = unbounded_id
        self.faces = {}
        self.edges = {}
#        self.stars = defaultdict(list)
        self.nodes = {}
        self.face_hierarchy = {}
        self.edge_hierarchy = {}

        # indexed geometry


#        self.points = None
#        self.points_lst = None
#        self.vertex_edge = None

# from cyrtree import RTree
#        self.edge_rtree = RTree()


def angle(orig, dest):
    """Angle that edge segment from point a to b defines 
    """
    dx = dest[0] - orig[0]
    dy = dest[1] - orig[1]
    xeq = orig[0] == dest[0]
    yeq = orig[1] == dest[1]
    assert not (xeq and yeq), "{} {}".format(orig, dest)
    angle = math.atan2(dy, dx)  # [-pi, pi]
    while angle <= 0:
        angle += TAU
    return angle


def dist(pa, pb):
    dx = pb[0] - pa[0]
    dy = pb[1] - pa[1]
    return (dx ** 2 + dy ** 2) ** 0.5


def eps_for_edge_geometry(g):
    eps = sys.float_info.max
    if len(g) > 2:
        for j in range(1, len(g) - 1):
            i = j - 1
            k = j + 1
            area = abs(orient2d(g[i], g[j], g[k]) * 0.5)
            base = dist(g[i], g[k])
            if base == 0:
                continue
            height = area / (0.5 * base)
            # height = sys.float_info.max
            if height < eps:
                eps = height
    return eps


def retrieve(DATASET, SRID, unbounded_id):
    pp = PlanarPartition(unbounded_id)

    sql = (
        "select st_setsrid(st_extent(geometry)::geometry(polygon),"
        + str(SRID)
        + ") from "
        + DATASET
        + "_edge;"
    )
    with connection(True) as db:
        mbr_geometry, = db.record(sql)
        dataset_envelope = mbr_geometry.envelope
    # unbounded face, otherwise this face cannot be found
    pp.faces[unbounded_id] = Face(unbounded_id, dataset_envelope, None, set([]), {})
    pp.face_hierarchy[unbounded_id] = None

    #print(f'Alex test: faces: {pp.faces}')
    faces = pp.faces
    face_hierarchy = pp.face_hierarchy
    edge_hierarchy = pp.edge_hierarchy

    t0 = time.time()
    sql = (
        "select face_id::int, feature_class::int, mbr_geometry, pip_geometry from "
        + DATASET
        + "_face"
    )  # where imp_low = 0"

    #    sql = "select id::int, height::int, null as mbr_geometry, null as pip_geometry from "+DATASET+"_face" # where imp_low = 0"
    with connection(True) as db:
        for item in db.irecordset(sql):
            face_id, feature_class_id, mbr_geometry, pip_geometry, = item
            faces[face_id] = Face(
                face_id,
                mbr_geometry.envelope,  # FIXME store as box2d and map to Envelope at connection level ???
                #                                mbr_geometry,
                pip_geometry,
                set([]),
                {"feature_class_id": feature_class_id, "step_low": 0},
            )
            face_hierarchy[face_id] = None

    print(f"{time.time()-t0:.3f}s face retrieval: {len(faces)} faces")
    #print(f"Alex Test: {faces[1019]}")
    t0 = time.time()

    edges = pp.edges
    # distinct, as the drenthe dataset has edges twice (with multiple feature class)
    sql = (
        "select distinct edge_id::int, start_node_id::int, end_node_id::int, left_face_id::int, right_face_id::int, geometry from "
        + DATASET
        + "_edge order by edge_id"
    )  # where imp_low = 0"
    #    sql = "select id::int, startnode::int, endnode::int, leftface::int, rightface::int, geometry from "+DATASET+"_edge" # where imp_low = 0"

    #    from tmp_mut_kdtree import create as create_kdtree
    #    kdtree = create_kdtree(dimensions=2)

    #    from tmp_grid import Grid
    #    from math import ceil
    #    kdtree = Grid(sizes = (ceil(dataset_envelope.width/50.), ceil(dataset_envelope.height/50.)))

    ### get unique vertices by hashing the geometry
    pts = {}

    with connection(True) as db:
        for item in db.irecordset(sql):
            edge_id, start_node_id, end_node_id, left_face_id, right_face_id, geometry, = (
                item
            )
            #            pp.edge_rtree.add(edge_id, geometry.envelope)
            edges[edge_id] = Edge(
                edge_id,
                start_node_id,
                angle(geometry[0], geometry[1]),
                end_node_id,
                angle(geometry[-1], geometry[-2]),
                left_face_id,
                right_face_id,
                geometry,
                {"step_low": 0}
                # {'smooth': make_smooth_line(geometry)})
            )
            # check for dup points
            for j in range(1, len(geometry)):
                i = j - 1
                assert geometry[i] != geometry[j], geometry

            # DO_SIMPLIFY
            for pt in geometry:
                if (pt.x, pt.y) not in pts:
                    pts[(pt.x, pt.y)] = [edge_id]
                else:
                    pts[(pt.x, pt.y)].append(edge_id)

            # add the edge_ids to the edges sets of the faces
            # on the left and right of the edge
            faces[left_face_id].edges.add(edge_id)
            faces[right_face_id].edges.add(~edge_id)
            edge_hierarchy[edge_id] = None
    print(f"{time.time()-t0:.3f}s edge retrieval: {len(edges)} edges")
    print(f"Alex Test: {faces[1019]}")
    t0 = time.time()
    # DO_SIMPLIFY

    tree = QuadTree(
        [
            (dataset_envelope.xmin, dataset_envelope.ymin),
            (dataset_envelope.xmax, dataset_envelope.ymax),
        ],
        64,
    )
    for pt in pts.keys():
        tree.add(pt)
    pp.quadtree = tree
    print(f"{time.time()-t0:.3f}s quadtree construction")
    t0 = time.time()

    ct = 0
    for pt in pp.quadtree:
        ct += 1
    print(f"spatially indexed {ct} points (quadtree)")

    #    from .tmp_kdtree import create as create_kdtree
    ##    from .tmp_mut_kdtree import create as create_kdtree
    #    pp.kdtree = create_kdtree(point_list=pts.keys(), dimensions=2)

    #    pp.kdtree = None
    #    def output_points(pts, fh):
    #        for pt in pts:
    #            fh.write("POINT({0[0]} {0[1]})\n".format(pt))

    #    with open('/tmp/kdtree_pts.wkt', 'w') as fh:
    #        fh.write('wkt\n')
    #        pts = pp.kdtree.range_search( (float('-inf'), float('-inf')), (float('+inf'), float('+inf')) )
    ## output_points((node.data for node in pp.kdtree.inorder()), fh)
    ##        output_points(pts, fh)
    ##    raw_input('paused')

    # check if we did not mix up things
    for edge_id in edges:
        edge = edges[edge_id]
        assert edge_id in faces[edge.left_face_id].edges
        assert ~edge_id in faces[edge.right_face_id].edges
    # stars
#    stars = pp.stars
#    for edge_id in edges:
#        stars[edges[edge_id].start_node_id].append(edge_id)  # outgoing: +
#        stars[edges[edge_id].end_node_id].append(~edge_id)  # incoming: -

    # nodes 
    nodes = pp.nodes
    for edge_id in edges:
        edge = edges[edge_id]
        # start node
        if edge.start_node_id not in nodes: 
            nodes[edge.start_node_id] = Node(edge.start_node_id, edge.geometry[0], [])
        nodes[edge.start_node_id].star.append(edge_id)
        # end node
        if edge.end_node_id not in nodes: 
            nodes[edge.end_node_id] = Node(edge.end_node_id, edge.geometry[-1], [])
        nodes[edges[edge_id].end_node_id].star.append(~edge_id)

    # sort edges counter clockwise (?) around a node
    sort_on_angle = partial(get_correct_angle, edges=edges)
#    for node_id in stars.keys():
#        stars[node_id].sort(key=sort_on_angle)
    for node_id in nodes.keys():
        nodes[node_id].star.sort(key=sort_on_angle)
    print(f"{time.time()-t0:.3f}s node stars")

    t0 = time.time()
    # based on the wheels we find, we can obtain the size of the faces
    for face in faces.values():
        wheels = get_wheel_edges(face.edges, pp)
        rings = [get_geometry_for_wheel(wheel, pp) for wheel in wheels]
        rings = [(abs(ring.signed_area()), ring) for ring in rings]
        rings.sort(reverse=True, key=lambda x: x[0])
        area, largest_ring = rings[0]
        perimeter = largest_ring.length
        iso_perimetric_quotient = (4.0 * math.pi * area) / (perimeter * perimeter)

        # FIXME: should we subtract hole regions from the faces?
        face.info["area"] = area
        face.info["perimeter"] = perimeter
        face.info["ipq"] = iso_perimetric_quotient
        face.info["priority"] = face.info["area"] * face.info["ipq"]
    print(f"{time.time()-t0:.3f}s area calculation")
    # when we ask the rtree the whole domain, we should get all edges
    #    assert len(pp.edge_rtree.intersection(dataset_envelope)) == len(edges)

    # return the planar partition
    return pp


###def retrieve_indexed(DATASET, SRID, unbounded_id):
###    pp = PlanarPartition(unbounded_id)

###    sql = "select st_setsrid(st_extent(geometry)::geometry(polygon),"+str(SRID)+") from "+DATASET+"_edge;"
###    with connection(True) as db:
###        mbr_geometry, = db.record(sql)
###        dataset_envelope = mbr_geometry.envelope
###    # unbounded face, otherwise this face cannot be found
###    pp.faces[unbounded_id] = Face(unbounded_id, dataset_envelope, None, set([]), {})
###    pp.face_hierarchy[unbounded_id] = None

###    faces = pp.faces
###    face_hierarchy = pp.face_hierarchy
###    edge_hierarchy = pp.edge_hierarchy

###    t0 = time.time()
###    sql = "select face_id::int, feature_class::int, mbr_geometry, pip_geometry from "+DATASET+"_face" # where imp_low = 0"

####    sql = "select id::int, height::int, null as mbr_geometry, null as pip_geometry from "+DATASET+"_face" # where imp_low = 0"
###    with connection(True) as db:
###        for item in db.irecordset(sql):
###            face_id, feature_class_id, mbr_geometry, pip_geometry, = item
###            faces[face_id] = Face(face_id,
###                                  mbr_geometry.envelope, # FIXME store as box2d and map to Envelope at connection level ???
###    #                                mbr_geometry,
###                                  pip_geometry,
###                                  set([]),
###                                  {'feature_class_id': feature_class_id, 'step_low': 0})
###            face_hierarchy[face_id] = None

###    print(time.time()-t0, "face retrieval")
###    t0 = time.time()

###    edges = pp.edges
###    # distinct, as the drenthe dataset has edges twice (with multiple feature class)
###    sql = "select distinct edge_id::int, start_node_id::int, end_node_id::int, left_face_id::int, right_face_id::int, geometry from "+DATASET+"_edge order by edge_id" # where imp_low = 0"
####    sql = "select id::int, startnode::int, endnode::int, leftface::int, rightface::int, geometry from "+DATASET+"_edge" # where imp_low = 0"

####    from tmp_mut_kdtree import create as create_kdtree
####    kdtree = create_kdtree(dimensions=2)

####    from tmp_grid import Grid
####    from math import ceil
####    kdtree = Grid(sizes = (ceil(dataset_envelope.width/50.), ceil(dataset_envelope.height/50.)))

###
####### get unique vertices by hashing the geometry
####    points = {}
####    points_lst = []
####    vertex_edge = {}
####    #^^INDEXED_GEOM

###    with connection(True) as db:
###        for item in db.irecordset(sql):
###            edge_id, start_node_id, end_node_id, left_face_id, right_face_id, geometry, = item

###            #^^INDEXED_GEOM
####            geometry_indexed = []
####            for coord in geometry:
####                point = (coord.x, coord.y)
####                if point not in points:
####                    index = len(points_lst)
####                    points[point] = index
####                    points_lst.append(point)
####                else:
####                    index = points[point]
####                geometry_indexed.append(index)

####                if index not in vertex_edge:
####                    vertex_edge[index] = [edge_id]
####                else:
####                    vertex_edge[index].append(edge_id)
###            #^^INDEXED_GEOM

####            pp.edge_rtree.add(edge_id, geometry.envelope)
###            edges[edge_id] = Edge(edge_id,
###                                  start_node_id, angle(geometry[0], geometry[1]),
###                                  end_node_id, angle(geometry[-1], geometry[-2]),
###                                  left_face_id, right_face_id,
###                                  geometry,
###                                  #^^INDEXED_GEOM
####                                  array.array('I', geometry_indexed),  #unsigned int indices
###                                  {'step_low': 0}
###                                     #{'smooth': make_smooth_line(geometry)})
###                                    )
###    #        print edge_id
###    #        print as_table(edges[edge_id].info['smooth'])
###            # add the edge_ids to the edges sets of the faces
###            # on the left and right of the edge
###            faces[left_face_id].edges.add(edge_id)
###            faces[right_face_id].edges.add(~edge_id)
###            edge_hierarchy[edge_id] = None

###    del points
####    pp.points = points

###    #^^INDEXED_GEOM
####    pp.points_lst = points_lst
####    pp.vertex_edge = vertex_edge
###    #^^INDEXED_GEOM

###    print(("points", len(pp.points_lst)))
###    from .tmp_kdtree import KdTree
###    pp.kdtree = KdTree(points_lst)
####    from mut_kdtree import create as create_kdtree
####    pp.kdtree = create_kdtree(point_list=points_lst, dimensions=2)
####    pp.kdtree = None

#####    def output_points(pts, fh):
#####        for pt in pts:
#####            fh.write("POINT({0[0]} {0[1]})\n".format(pt))

#####    with open('/tmp/kdtree_pts.wkt', 'w') as fh:
#####        fh.write('wkt\n')
#####        pts = pp.kdtree.range_search( (float('-inf'), float('-inf')), (float('+inf'), float('+inf')) )
#####        ## output_points((node.data for node in pp.kdtree.inorder()), fh)
#######        output_points(pts, fh)
#######    raw_input('paused')

###    print(time.time()-t0, "edge retrieval")
###    t0 = time.time()

###    # check if we did not mix up things
###    for edge_id in edges:
###        edge = edges[edge_id]
###        assert edge_id in faces[edge.left_face_id].edges
###        assert ~edge_id in faces[edge.right_face_id].edges
###    # stars
###    stars = pp.stars
###    for edge_id in edges:
###        stars[edges[edge_id].start_node_id].append(edge_id)  # outgoing: +
###        stars[edges[edge_id].end_node_id].append(~edge_id)   # incoming: -

###    # sort edges counter clockwise (?) around a node
###    sort_on_angle = partial(get_correct_angle, edges=edges)
###    for node_id in stars.keys():
###        stars[node_id].sort(key=sort_on_angle)

###    print(time.time()-t0, "node stars")
###    t0 = time.time()

###    t0 = time.time()
###    # based on the wheels we find, we can obtain the size of the faces
###    for face in faces.values():
###        wheels = get_wheel_edges(face.edges, pp)
###        rings = [get_geometry_indexed_for_wheel(wheel, pp) for wheel in wheels]
###        rings = [(abs(ring.signed_area()), ring) for ring in rings]
####        rings.sort(reverse = True, key=lambda x: x[0])
####        area, largest_ring = rings[0]
###        area, largest_ring = max(rings, key=lambda x: x[0])
###        perimeter = largest_ring.length

###        iso_perimetric_quotient = (4.0 * math.pi * area) / (perimeter * perimeter)

###        # FIXME: should we subtract hole regions from the faces?
###        face.info['area'] = area
###        face.info['perimeter'] = perimeter
###        face.info['ipq'] = iso_perimetric_quotient
###        face.info['priority'] = face.info['area'] * face.info['ipq']
###    print(time.time()-t0, "area calculation")

###    # when we ask the rtree the whole domain, we should get all edges
####    assert len(pp.edge_rtree.intersection(dataset_envelope)) == len(edges)

###    # return the planar partition
###    return pp


def do_output_quadtree(pp):
    print("OUTPUTTING quadTREE")
    from .tmp_simplify import output_points

    with open("/tmp/quadtree_pts.wkt", "w") as fh:
        fh.write("wkt\n")
        output_points((pt for pt in pp.quadtree), fh)


def dissolve_unwanted_nodes(pp):
    """
    Modify planar partition by dissolving edges that share a node
    of degree 2 where this is not needed
    """
    new_edge_id = max(pp.edges.keys()) + 1
    dissolve = []
    for node_id, node in pp.nodes.items():
        star = node.star
#    for node_id, star in pp.stars.items():
        if len(star) == 2 and len(set(map(positive_id, star))) == 2:
            #print(f"Alex Test: node dissolved: {node_id}, {node}")
            pair = tuple(map(positive_id, star))
            dissolve.append((node_id, pair))

    for new_edge_id, (node_id, pair) in enumerate(dissolve, start=new_edge_id):
        merge_edge_pair(pair, node_id, new_edge_id, pp, None, None, 0)


#        do_output_quadtree(pp)
#        input('paused - check after merge pair')


def parent(par, hierarchy, shorten=False):
    """Look up the parent of an object in the hierarchy dict

    We *could* shorten the path in the hierarchy afterwards 
    (with a recursive algorithm this could be on the way out)
    """
    # seen = []
    init = par
    try:
        while par is not None:
            last = par
            # seen.append(last)
            par = hierarchy[par]
    except KeyError:
        raise KeyError("while looking up {} I did not find {}".format(init, par))
    # update the hierarchy to shorten paths to traverse in the future
    # FIXME: WOULD THIS HELP PERFORMANCE A LOT? NOTE: depends on whether info is still needed for writing to DB whether this *CAN* be done!
    # -- modifies hierarchy
    #    if shorten and seen:
    #        for val in seen[:-1]:
    #            hierarchy[val] = seen[-1]
    return last


# -- make node stars (how edges are sorted geometrically around a node)

# FIXME: nodes with a single incident edge
# you will (most likely) not visit both sides of the edge automatically
# two options: a. leave as is, and do not care;
#              b. add 1-complement of edge_id to the list


def get_correct_angle(signed_edge_id, edges):
    if signed_edge_id < 0:
        return edges[~signed_edge_id].end_angle
    else:
        return edges[signed_edge_id].start_angle


# FIXME: unify as one method that takes boolean argument ccw=True?
def get_ccw_face(signed_edge_id, edges, face_hierarchy):
    if signed_edge_id < 0:  # incoming edge
        face_id = edges[~signed_edge_id].right_face_id
    else:  # outgoing edge
        face_id = edges[signed_edge_id].left_face_id
    return parent(face_id, face_hierarchy)


def get_cw_face(signed_edge_id, edges, face_hierarchy):
    if signed_edge_id < 0:  # incoming edge
        face_id = edges[~signed_edge_id].left_face_id
    else:  # outgoing edge
        face_id = edges[signed_edge_id].right_face_id
    return parent(face_id, face_hierarchy)


# -- find order for visiting a loop
# we can make a function here that transform the edges list of the face into
# wheels
# (list with lists of sorted edge ids in correct order for making a ring)
# if we would have parallelism, this could run in parallel easily
# (only globally shared read only information)
def get_wheel_edges(unsorted_edges_for_face, pp):
    edges = pp.edges
#    stars = pp.stars
    nodes = pp.nodes
    # do copy input, makes function side effect free
    unsorted_edges = unsorted_edges_for_face.copy()
    edge_id = None
    wheels = []
    #    print "unsorted edges", unsorted_edges_for_face
    while unsorted_edges:
        if edge_id is None:
            wheel = []
            edge_id = unsorted_edges.pop()
        else:
            unsorted_edges.remove(edge_id)
        wheel.append(edge_id)
        if edge_id > 0:
            node_id = edges[edge_id].end_node_id
        else:
            node_id = edges[~edge_id].start_node_id
#        star = stars[node_id]
        star = nodes[node_id].star
        angles = [get_correct_angle(_, edges) for _ in star]
        indx = star.index(~edge_id)
        next_edge_id = star[(indx - 1) % len(star)]
        #        print ""
        #        print " edge_id", edge_id, " [~", ~edge_id, "]"
        #        print "  @-node", node_id
        #        print "  star ", star
        #        print "  star+", map(positive_id, star)
        #        print "  angles", angles
        #        print "  indx", indx
        #        print "  next edge in wheel", next_edge_id, next_edge_id in wheels
        if next_edge_id in unsorted_edges:
            edge_id = next_edge_id
        else:
            wheels.append(wheel)
            edge_id = None
    return wheels


# -- calculate for each face its area
def is_ccw(signed_area):
    """Returns True when a ring is oriented counterclockwise

    This is based on the signed area:

     > 0 for counterclockwise
     = 0 for none (degenerate)
     < 0 for clockwise
    """
    if signed_area > 0:
        return True
    elif signed_area < 0:
        return False
    else:
        raise ValueError("Degeneracy: No orientation based on area")


# -- functions for getting the next index in the list, dependent on orientation
cur_pos = lambda x: x
nxt_pos = lambda x: x + 1
cur_neg = lambda x: ~x
nxt_neg = lambda x: ~x - 1


def get_area(wheels, pp):
    """Calculates area, does visit all coordinates of all edges of the face
    """
    edges = pp.edges
    area = 0
    for wheel in wheels:  # FIXME: check correct orientation?
        wheel_area = 0
        for edge_id in wheel:
            # take edge in correct direction
            # FIXME:
            # IS THIS DIRECTIONALITY REALLY NEEDED FOR CORRECT area calculation?
            # or can you just sum up all parts under a segment?
            edge = edges[edge_id if edge_id > 0 else ~edge_id]
            geom = edge.geometry
            ct = len(geom)
            # dependent on whether we go in the correct direction
            # take the function that gives back the right
            # index into the edge geometry for the current coordinate
            # and the next coordinate
            if edge_id > 0:
                cur = cur_pos
                nxt = nxt_pos
            else:
                cur = cur_neg
                nxt = nxt_neg
            # NOTE: this possibly can overflow if xy-values are rather large
            # this could be solved by shifting the geometry towards origin
            # of the domain, or take the very first coordinate and subtract
            # this xy value from all others (to make local coordinate system)
            for i in range(ct - 1):
                c = cur(i)
                n = nxt(i)
                dx = geom[n][0] - geom[c][0]  # x
                dy = geom[n][1] + geom[c][1]  # y
                wheel_area += dx * dy
        wheel_area *= 0.5
        area += wheel_area
    return abs(area)  # FIXME


def get_geometry_for_wheel(wheel, pp):
    edges = pp.edges
    ln = None
    for edge_id in wheel:
        # take edge geometry in correct direction
        edge = edges[edge_id if edge_id > 0 else ~edge_id]
        g = edge.geometry[:]
        if edge_id < 0:
            g.reverse()
        if ln is None:
            # if this is the first edge of the wheel take it completely
            ln = g
        else:
            # otherwise,
            # take off last coordinate
            # and add linestring to it
            ln.pop()
            ln.extend(g)
    return simplegeom.geometry.LinearRing(ln)


def get_geometry_indexed_for_wheel(wheel, pp):
    edges = pp.edges
    ln = None
    for edge_id in wheel:
        # take edge geometry in correct direction
        edge = edges[edge_id if edge_id > 0 else ~edge_id]
        g = edge.geometry[:]
        if edge_id < 0:
            g.reverse()
        if ln is None:
            # if this is the first edge of the wheel take it completely
            ln = g
        else:
            # otherwise,
            # take off last coordinate
            # and add linestring to it
            ln.pop()
            ln.extend(g)
    coords = [pp.points_lst[indx] for indx in ln]
    return simplegeom.geometry.LinearRing(coords)


def find_neighbours(face_id, pp):
    """Generator for neighbour face ids
    """
    face_hierarchy = pp.face_hierarchy
    faces = pp.faces
    edges = pp.edges
    #    for edge_id in faces[face_id].edges:
    #        edge = edges[edge_id if edge_id > 0 else ~edge_id]
    #        if parent(edge.left_face_id, face_hierarchy) == face_id:
    #            neighbour_id = parent(edge.right_face_id, face_hierarchy)
    #        else:
    #            assert parent(edge.right_face_id, face_hierarchy) == face_id
    #            neighbour_id = parent(edge.left_face_id, face_hierarchy)
    #        if neighbour_id != face_id:
    #            yield neighbour_id
    # get positive edge ids
    it1 = [edges[edge_id] for edge_id in map(positive_id, faces[face_id].edges)]
    # for each edge translate its left and right face and return this pair
    it2 = [
        (
            parent(edge.left_face_id, face_hierarchy, False),
            parent(edge.right_face_id, face_hierarchy, False),
        )
        for edge in it1
    ]
    # return the other face of the pair as neighbour
    # each neighbour face id will only be returned once (due to conversion to set)
    it3 = [pair[1] if pair[0] == face_id else pair[0] for pair in it2]
    return set(it3)
    # this gives multiple ones for neighbours
    ## return it3
    ## allows universe to be selected, even if more direct neighbours -> edge problem???


def common_boundary(face_id, neighbour_id, pp):
    """Find the list of edge ids that are between two faces.
    
    Note that the edge ids returned do not represent any particular direction.
    """
    faces = pp.faces
    edges = pp.edges
    face_hierarchy = pp.face_hierarchy

    #    face = faces[face_id]
    #    neighbour = faces[neighbour_id]
    #    # walk over the face with the least amount of edges
    #    if len(neighbour.edges) <= len(face.edges):
    #        edge_ids = face.edges
    #    else:
    #        edge_ids = neighbour.edges
    #    for edge_id in map(positive_id, edge_ids):
    #        edge = edges[edge_id]
    #        if (parent(edge.left_face_id, face_hierarchy) == face_id and parent(edge.right_face_id, face_hierarchy) == neighbour_id) or \
    #            (parent(edge.left_face_id, face_hierarchy) == neighbour_id and parent(edge.right_face_id, face_hierarchy) == face_id):
    #            yield edge.id

    face = faces[face_id]
    neighbour = faces[neighbour_id]
    face_edges = set(map(positive_id, face.edges))
    neighbour_edges = set(map(positive_id, neighbour.edges))
    return face_edges.intersection(neighbour_edges)


# -- modifier functions
# remove a edge
def remove_edge(edge_id, pp, edge_seq):
    """
    """
    # what we have to update for removal of an edge:
    #
    # the set of the unordered edges of every face that has a relationship with the edge
    # faces[edge.left_face_id].edges
    # faces[edge.right_face_id].edges
    #
    # the node stars
    # stars[edges[edge_id].start_node_id].append(edge_id) # outgoing
    # stars[edges[edge_id].end_node_id].append(~edge_id) # incoming
    #
    # the edges{} dictionary
    #
    # Note:
    # After the removal, the wheels of the two incident faces are not really
    # correct wheels any more, i.e. they do not form a complete wheel
    # because removing an edge means unifying the two adjacent faces
    # that should be handled by the caller of this method
    edges = pp.edges
    face_hierarchy = pp.face_hierarchy
    faces = pp.faces
    nodes = pp.nodes
#    stars = pp.stars
    #    edge_rtree = pp.edge_rtree
    #    present_ids = edge_rtree.intersection(edges[edge_id].geometry.envelope)
    #    if edge_id in present_ids:
    #        print('deleting', edge_id)
    #        edge_rtree.delete(edge_id, edges[edge_id].geometry.envelope)

    # take complement of edge identifier if needed
    edge_id = positive_id(edge_id)
    # remove the edge from the edge dictionary
    edge = edges.pop(edge_id)

    # remove the edge from the edges list of the face
    faces[parent(edge.left_face_id, face_hierarchy)].edges.remove(edge_id)
    faces[parent(edge.right_face_id, face_hierarchy)].edges.remove(~edge_id)
    # remove from stars list
#    stars[edge.start_node_id].remove(edge_id)  # outgoing
#    stars[edge.end_node_id].remove(~edge_id)  # incoming

    nodes[edge.start_node_id].star.remove(edge_id)  # outgoing
    nodes[edge.end_node_id].star.remove(~edge_id)  # incoming

    # FIXME: Update stars / Remove star when its edge list has become 0 length?

    # DO_SIMPLIFY
    ##

    # pp.kdtree =
    #        print('removing;POINT({0} {1})'.format(pt.x, pt.y))

    # remove intermediate points from the quadtree
    # (end points of the polyline will be removed when 
    # start/end node will be removed)
    first, last = 0, len(edge.geometry) - 1
    for i, pt in enumerate(edge.geometry):
        if i == first or i == last:
            continue
        pp.quadtree.remove((pt.x, pt.y))
    #        is_deleted = pp.kdtree.delete((pt.x, pt.y))
    #        assert is_deleted
    ##
    #    for idx, node_id in [(first, edge.start_node_id), (last, edge.end_node_id)]:
    #        if len(stars[node_id]) == 0:
    #            pt = edge.geometry[idx]
    #            pp.kdtree = pp.kdtree.remove((pt.x, pt.y))
    ##

    # DO_SIMPLIFY
    # remove the edge from the edge sequence
    # print('removing', edge_id)
    if edge_seq is not None:
        edge_seq.pop(edge_id)

    return (edge.start_node_id, edge.end_node_id)


def remove_node(node_id, pp):
    """
    """
    nodes = pp.nodes
    node = nodes.pop(node_id)
    # also remove the point from the quadtree
    pt = node.geometry
    pp.quadtree.remove((pt.x, pt.y))
#    stars = pp.stars
#    stars.pop(node_id)

    # FIXME: store node


def remove_face(face_id, pp):
    """
    """
    pp.faces.pop(face_id)


def edge_pairs(node_ids, pp):
    """
    """
    # - = incoming , + = outgoing edges around node
    # 4 permutations:
    # --
    # +-
    # -+
    # ++
#    stars = pp.stars
    nodes = pp.nodes
    pairs = []
    for node_id in node_ids:
        if len(nodes[node_id].star) == 2:
            # FIXME: should we keep the edge ids signed,
            # so we know the direction around the nodes??
            edge_id_pair = (tuple(map(positive_id, nodes[node_id].star)), node_id)
            pairs.append(edge_id_pair)
    return pairs


# FIXME: rename as 'store_edge' ?
def output_edge(output, pp, edge_id, face_step):
    """output a edge to the data storage
    """
    # START
    #        [edge_id,
    #         start, end,
    #         left_low, right_low, left_high, right_high,
    #         imp_low, imp_high,
    #         step_low, step_high,
    #         edge_class,
    #         pickled_blg,
    #         smoothline,
    #         path],
    if not output:
        return
    e = pp.edges[edge_id]
    if e.info["step_low"] == face_step:
        #        print('skipping output for edge {}'.format(edge_id))
        return
    assert face_step > e.info["step_low"], "id: {}, step-lo: {}, step-hi: {}".format(
        e.id, e.info["step_low"], face_step
    )
    tup = (
        e.id,                                       # edge id
        e.info["step_low"],                         # step low
        face_step,                                  # step high

        e.start_node_id,                            # start node
        e.end_node_id,                              # end node
        e.left_face_id,                             # left face low
        e.right_face_id,                            # right face low
        parent(e.left_face_id, pp.face_hierarchy),  # left face high
        parent(e.right_face_id, pp.face_hierarchy), # right face high
        #0,
        #0,
        None,                                       # edge klass (when it represents a linear object)
        #None,
        #None,
        e.geometry,                                 # geometry
    )
    output.edge.append(*tup)


# END


def merge_edge_pair(pair, middle_node_id, new_edge_id, pp, edge_seq, output, face_step):
    """
    pair                2-tuple with edge ids
    middle_node_id      the node id in the middle
    """
    edge_hierarchy = pp.edge_hierarchy
    face_hierarchy = pp.face_hierarchy
    edges = pp.edges
    faces = pp.faces
#    stars = pp.stars
    nodes = pp.nodes

    one_id, other_id = pair
    one_id = parent(one_id, edge_hierarchy)
    other_id = parent(other_id, edge_hierarchy)
    if one_id == other_id:
        # print ("> skipping edge merge", one_id, "with itself", other_id)
        return
    one = edges[one_id]
    other = edges[other_id]
    if False:
        print(
            f"   > merging edge pair {one_id}, {other_id} @ {middle_node_id} into {new_edge_id}"
        )
    #    print (one.geometry)
    #    print (other.geometry)
    order = 0
    if one.end_node_id == middle_node_id:
        # correct
        start_node_id = one.start_node_id
        middle_node_id = one.end_node_id
        left_face_id = parent(one.left_face_id, face_hierarchy)
        right_face_id = parent(one.right_face_id, face_hierarchy)
        order += 0
    else:
        # flip
        start_node_id = one.end_node_id
        middle_node_id = one.start_node_id
        left_face_id = parent(one.right_face_id, face_hierarchy)
        right_face_id = parent(one.left_face_id, face_hierarchy)
        order += 1

    if other.end_node_id == middle_node_id:
        # flip
        end_node_id = other.start_node_id
        assert other.end_node_id == middle_node_id
        assert left_face_id == parent(other.right_face_id, face_hierarchy), f"{left_face_id} vs {parent(other.right_face_id, face_hierarchy)}"
        assert right_face_id == parent(other.left_face_id, face_hierarchy), f"{right_face_id} vs {parent(other.left_face_id, face_hierarchy)}"
        order += 4
    else:
        # correct
        end_node_id = other.end_node_id
        assert other.start_node_id == middle_node_id
        assert left_face_id == parent(other.left_face_id, face_hierarchy)
        assert right_face_id == parent(other.right_face_id, face_hierarchy)
        order += 2

    if False:
        print(
            f"     @ nodes: {start_node_id}, {middle_node_id}, {end_node_id} order: {order}"
        )

    if order == 4:
        geometry = one.geometry[:-1]
        geometry.extend(other.geometry[len(other.geometry) - 1 :: -1])
    elif order == 2:
        geometry = one.geometry[:]
        geometry.extend(other.geometry[1:])
    elif order == 3:
        geometry = one.geometry[::-1]
        geometry.extend(other.geometry[1:])
    elif order == 5:
        geometry = one.geometry[::-1]
        geometry.pop()
        geometry.extend(other.geometry[len(other.geometry) - 1 :: -1])

    for j in range(1, len(geometry)):
        i = j - 1
        assert geometry[i] != geometry[j], geometry

    # update the planar partition
    for tmp_id in (one_id, other_id):
        if output is not None:
            output_edge(output, pp, tmp_id, face_step)
        remove_edge(tmp_id, pp, edge_seq)
    del tmp_id
    #
    remove_node(middle_node_id, pp)

    #    vertex_check(pp)
    # make new edge
    edge_id = new_edge_id
    # add it to the planar partition
    edges[edge_id] = Edge(
        edge_id,
        start_node_id,
        angle(geometry[0], geometry[1]),
        end_node_id,
        angle(geometry[-1], geometry[-2]),  # angle from last point to second last point
        left_face_id,
        right_face_id,
        geometry,
        {"step_low": face_step},
    )  # FIXME: Keep info on merged edge?

    # DO_SIMPLIFY
    for pt in geometry:
        #        print('adding;POINT({0} {1})'.format(pt.x, pt.y ) )
        pp.quadtree.add((pt.x, pt.y))
    #        is_found = pp.kdtree.undelete([pt.x ,pt.y]) # FIXME: first/last point different?
    #        assert is_found
    #    vertex_check(pp)
    # update faces
    faces[parent(left_face_id, face_hierarchy)].edges.add(edge_id)
    faces[parent(right_face_id, face_hierarchy)].edges.add(~edge_id)
    # update stars
#    stars[edges[edge_id].start_node_id].append(edge_id)  # outgoing: +
#    stars[edges[edge_id].end_node_id].append(~edge_id)  # incoming: -
    nodes[edges[edge_id].start_node_id].star.append(edge_id)  # outgoing: +
    nodes[edges[edge_id].end_node_id].star.append(~edge_id)  # incoming: -

    sort_on_angle = partial(get_correct_angle, edges=edges)
    for tmp_id in [start_node_id, end_node_id]:
#        stars[tmp_id].sort(key=sort_on_angle)
        nodes[tmp_id].star.sort(key=sort_on_angle)
    del tmp_id
    # add merge to edge_hierarchy
    edge_hierarchy[one_id] = edge_id
    edge_hierarchy[other_id] = edge_id
    edge_hierarchy[edge_id] = None

    # DO_SIMPLIFY
    # print('made new edge', edge_id)
    if edge_seq is not None:
        eps = eps_for_edge_geometry(geometry)
        edge_seq[edge_id] = eps


#        if eps == 0:
#            # FIXME: intermediate points on a straight line
#            # should be removed directly, as they do not bring much
# (except that the edge will have to be simplified next)!
#            print(f"not nice eps for edge id {edge_id}")


def output_face(output, pp, face_id, face_step):
    """output a face to the data storage
    """
    # START
    #        [face_id,
    #         imp_low, imp_high,
    #         step_low, step_high,
    #         imp_own,
    #         area,
    #         klass,
    #         mbr, pip],
    if not output:
        return
    f = pp.faces[face_id]
    tup = (
        f.id,
        #None,
        #None,
        f.info["step_low"],
        face_step,
        #None,
        f.info["area"],
        f.info["feature_class_id"],
        f.mbr_geometry,
        f.pip_geometry,
    )
    output.face.append(*tup)


# END


def output_face_hierarchy(output, pp, face_id, parent_face_id, face_step):
    """output a face with its parent to the data storage
    """
    # START
    #     [face_id,
    #      imp_low, imp_high,
    #      step_low, step_high,
    #      parent_id,
    #      ],
    if not output:
        return

    f = pp.faces[face_id]
    assert f.id == face_id
    tup = (f.id, 
            #None, None, 
            f.info["step_low"], face_step, parent_face_id)
    output.face_hierarchy.append(*tup)


def universe_merge_face_pair(
    face_id, neighbour_face_id, new_face_id, pp, edge_seq, output, face_step
):
    """
    merge a face into the universe face (the neighbour)
    """
    assert neighbour_face_id == pp.unbounded_id
    # print "merging", face_id, "into the universe ", pp.unbounded_id
    faces = pp.faces
    face_hierarchy = pp.face_hierarchy
#    stars = pp.stars
#    nodes = pp.nodes

    nodes_to_check = []
    common = list(common_boundary(face_id, neighbour_face_id, pp))
    common_boundary_length = sum(
        (pp.edges[edge_id].geometry.length for edge_id in common)
    )
    # if len(common) > 10:
    #    print len(common),"*"
    for edge_id in common:
        output_edge(output, pp, edge_id, face_step)
        nodes_to_check.extend(remove_edge(edge_id, pp, edge_seq))
        # we merge edges and their geometries into longer chains
        # we get a node_id list back from the remove_edge function
        # then we can check whether at these nodes there is exactly
        # two edges incident and we subsequently glue the edges in a binary
        # fashion

    #            # we could make a set of all glue_edges (positive ids)
    #            # start at one of these and then walk in both directions
    #            for edge_id in glue_edges:
    #                if edge_id > 0:
    #                    node_id = edges[edge_id].end_node_id
    #                else:
    #                    node_id = edges[~edge_id].start_node_id
    #                star = stars[node_id]
    #                indx = star.index(~edge_id) # reverse edge direction and see where it is
    #                next_edge_id = star[(indx-1) % len(star)]
    #            #print "TODO: union edges around node with id", node_id

    # Make a new Face object replacing the two old faces...
    # now we should union the mbr_geometry
    #    mbr = faces[face_id].mbr_geometry
    #    mbr.enlarge_by(faces[neighbour_face_id].mbr_geometry)
    # and we should pick one pip_geometry (of the neighbour)
    #    pip = faces[neighbour_face_id].pip_geometry
    # and we should pick a new feature_class (of the neighbour)
    #    info = faces[neighbour_face_id].info.copy()
    #    info['area'] += faces[face_id].info['area']
    #    info['perimeter'] += faces[face_id].info['perimeter']
    #    info['perimeter'] -= 2.0 * common_boundary_length
    #    info['step_low'] = face_step
    #    # and then we can make one new face...
    #    new_face = Face(new_face_id,
    #            mbr,
    #            pip,
    #            # union the edge lists of the two faces
    #            faces[face_id].edges.union(faces[neighbour_face_id].edges),
    #            info
    #    )
    #    assert new_face_id not in faces
    #    # that we add to the faces dictionary
    #    faces[new_face_id] = new_face

    # add info about the merge into the face face_hierarchy dict
    face_hierarchy[face_id] = pp.unbounded_id
    #    face_hierarchy[neighbour_face_id] = new_face_id
    #    face_hierarchy[new_face_id] = None

    output_face(output, pp, face_id, face_step)
    #    output_face(output, pp, neighbour_face_id, face_step)

    # FIXME: OUTPUT face_hierarchy
    output_face_hierarchy(output, pp, face_id, pp.unbounded_id, face_step)
    #    output_face_hierarchy(output, pp, neighbour_face_id, new_face_id, face_step)

    # we remove the two old faces
    remove_face(face_id, pp)
    #    remove_face(neighbour_face_id, pp)

    return nodes_to_check


# -- merge 2 faces into new face
def merge_face_pair(
    face_id, neighbour_face_id, new_face_id, pp, edge_seq, output, face_step
):
    """
    """
    # print "merging", face_id, "to", neighbour_face_id, "will become", new_face_id
    faces = pp.faces
    face_hierarchy = pp.face_hierarchy
#    stars = pp.stars
    nodes = pp.nodes

    nodes_to_check = []
    common = list(common_boundary(face_id, neighbour_face_id, pp))
    common_boundary_length = sum(
        (pp.edges[edge_id].geometry.length for edge_id in common)
    )
    # if len(common) > 10:
    #    print len(common),"*"
    for edge_id in common:
        if False:
            print(f"   _ removing edge {edge_id}")
        output_edge(output, pp, edge_id, face_step)
        nodes_to_check.extend(remove_edge(edge_id, pp, edge_seq))
        # we merge edges and their geometries into longer chains
        # we get a node_id list back from the remove_edge function
        # then we can check whether at these nodes there is exactly
        # two edges incident and we subsequently glue the edges in a binary
        # fashion

    #            # we could make a set of all glue_edges (positive ids)
    #            # start at one of these and then walk in both directions
    #            for edge_id in glue_edges:
    #                if edge_id > 0:
    #                    node_id = edges[edge_id].end_node_id
    #                else:
    #                    node_id = edges[~edge_id].start_node_id
    #                star = stars[node_id]
    #                indx = star.index(~edge_id) # reverse edge direction and see where it is
    #                next_edge_id = star[(indx-1) % len(star)]
    #            #print "TODO: union edges around node with id", node_id

    # Make a new Face object replacing the two old faces...
    # now we should union the mbr_geometry
    mbr = faces[face_id].mbr_geometry
    mbr.enlarge_by(faces[neighbour_face_id].mbr_geometry)
    # and we should pick one pip_geometry (of the neighbour)
    pip = faces[neighbour_face_id].pip_geometry
    # and we should pick a new feature_class (of the neighbour)
    info = faces[neighbour_face_id].info.copy()
    info["area"] += faces[face_id].info["area"]
    info["perimeter"] += faces[face_id].info["perimeter"]
    info["perimeter"] -= 2.0 * common_boundary_length
    info["step_low"] = face_step
    # and then we can make one new face...
    new_face = Face(
        new_face_id,
        mbr,
        pip,
        # union the edge lists of the two faces
        faces[face_id].edges.union(faces[neighbour_face_id].edges),
        info,
    )
    assert new_face_id not in faces
    # that we add to the faces dictionary
    faces[new_face_id] = new_face

    # add info about the merge into the face face_hierarchy dict
    face_hierarchy[face_id] = new_face_id
    face_hierarchy[neighbour_face_id] = new_face_id
    face_hierarchy[new_face_id] = None

    output_face(output, pp, face_id, face_step)
    output_face(output, pp, neighbour_face_id, face_step)

    # FIXME: OUTPUT face_hierarchy
    output_face_hierarchy(output, pp, face_id, new_face_id, face_step)
    output_face_hierarchy(output, pp, neighbour_face_id, new_face_id, face_step)

    # we remove the two old faces
    remove_face(face_id, pp)
    remove_face(neighbour_face_id, pp)

    return nodes_to_check


if __name__ == "__main__":
    # DATASET, unbounded_id = "tp_toponl", 0
    DATASET, unbounded_id, SRID = "top10nl_drenthe", 0, 28992
    retrieve_indexed(DATASET, SRID, unbounded_id)
##    retrieve(DATASET, SRID, unbounded_id)
