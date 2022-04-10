# encoding: utf-8

## START print replacement
## replacement for print function, to have also the file and linenumber printed where print is being invoked
## warning: if print is used for writing to a file object, this messes it up (kwargs not taken into account)
# from inspect import getframeinfo, stack
# import sys
# def print(*args, **kwargs):
#    message = " ".join(map(str, args))      # <-- stack()[1][0] for this line
#    caller = getframeinfo(stack()[1][0])
#    sys.stdout.write("%s:%d - %s\n" % (caller.filename.split("/")[-1], caller.lineno, message)) # python3 syntax print
#    sys.stdout.flush()
# import builtins
# builtins.print = print
## END print replacement
import logging
import sys

import matplotlib
from geopandas import geoseries, GeoSeries #module & class?
from shapely import errors as shpErr, geometry, wkt

# logging is used inside connection + grassfire (split), we can set the level here

logging.basicConfig(level=logging.FATAL)
import pprint

from connection import connection
import simplegeom.geometry
import math
import array
import time
import sys
from tri.delaunay import triangulate, ToPointsAndSegments
from tri.delaunay.iter import (
    InteriorTriangleIterator,
    FiniteEdgeIterator,
    RegionatedTriangleIterator,
)
from tri.delaunay.inout import output_triangles, output_edges, output_vertices
from . import pqdict
from . import scalestep
from .datastructure import (
    retrieve,
    dissolve_unwanted_nodes,
    eps_for_edge_geometry,
    Node,
    Edge,
    Face,
    angle,
    find_neighbours,
    common_boundary,
    merge_face_pair,
    universe_merge_face_pair,
    edge_pairs,
    merge_edge_pair,
    output_edge,
    output_face,
    output_face_hierarchy,
    output_pp_wkt,
    positive_id,
    get_correct_angle,
    get_ccw_face,
    get_cw_face,
    parent,
    remove_edge,
    remove_face,
    remove_node,
)
from functools import partial
from splitarea.harvest import (
    EdgeEdgeHarvester,
    MidpointHarvester,
    VertexInfo,
    ConnectorPicker,
)
from splitarea.skeleton import (
    make_graph,
    label_sides,
    prune_branches,
    define_groups,
    make_new_node_ids,
    make_new_edges,
)
from .schema import output_layers
from .loader import AsyncLoader
from io import StringIO
from contextlib import closing

# FIXME: make possible to use both, dependent on line?
# from .edge_simplify.visvalingham_whyatt import simplify
#from .edge_simplify.reumann_witkam import simplify_reumann_witkam as simplify
from .edge_simplify.samsonov_yakimova2 import simplifySY as simplify

import sys



# FIXME:
# - these parameters should be either program arguments
# - or could come from metadata table in DBMS (better)


# DATASET, unbounded_id = "gima_goes", -1

# DATASET, unbounded_id = "tp_toponl", 0
# SRID = 28992
# BASE_DENOMINATOR = 10000

# DATASET, unbounded_id = "top10nl_drenthe", 0
# SRID = 28992
# BASE_DENOMINATOR = 10000

DATASET, unbounded_id = "top10nl_limburg_tiny", 0
SRID = 28992
BASE_DENOMINATOR = 10000

# DATASET, unbounded_id = "clc_est", 0
# SRID = 3035
# BASE_DENOMINATOR = 100000

# DATASET, unbounded_id = "atkis_univ", 1
# SRID = 32632
# BASE_DENOMINATOR = 50000

OUTPUT_DATASET_NM = DATASET

# whether to perform line simplification on the edge geometries
do_edge_simplification = True

# whether to show detailed progress information
# -- shows detailed progress on which face / edge is dealt with
do_show_progress = False  # rename to do_show_trace?

# whether to use the straight skeleton code as backend to generate new boundaries while splitting areas
# -- FIXME: not operational yet (ids of nodes)
do_use_grassfire = False

# whether to check the edge geometries (polylines) for (self-) intersection during the process
do_expensive_post_condition_check_simplify = False
do_expensive_check_after_each_simplification = True

# some stats on what type of vertices are created while splitting areas
STATS_SPLIT_VERTEX_TYPES = {0: 0, 1: 0, 2: 0, 3: 0}


print(("Processing {}".format(DATASET)))

# -- greedy algorithm for simplifying
# pick a face and random neighbour, then merge them
# output_wkt('pre')

# def face_compare(x, y):
#    """
#    Return negative if x<y, zero if x==y, positive if x>y.
#    """
#    if x.info['area'] > y.info['area']:
#        return 1
#    elif x.info['area'] == y.info['area']:
#        if id(x) < id(y):
#            return -1
#        elif id(x) > id(y):
#            return 1
#        else:
#            return 0
#    else:
#        return -1


# def points_segments(rings):
#    """
#    """
#    points = []
#    segments = []
#    points_idx = {}
#    for ring in rings:
#        for pt in ring[:-1]:
#            if pt not in points_idx:
#                points_idx[pt] = len(points)
#                points.append(pt)
#        for start, end in zip(ring, ring[1:]):
#            segments.append((points_idx[start], points_idx[end]))
#    return points, segments


# def index_edges_for_face(face, pp):
#    from cyrtree import RTree
##    print("")
##    print(face.id, len(face.edges))
#    if len(face.edges) > 10:
#        tree = RTree()
#        for signed_edge_id in face.edges:
#            edge_id = positive_id(signed_edge_id)
#            edge = pp.edges[edge_id]
#            print(edge.geometry.envelope)
#            tree.add(edge_id, edge.geometry.envelope)
#        face.info['rtree'] = tree


# def get_overlapping_edges(envelope, face, pp):
##    if 'rtree' in face.info:
##        # search rtree of the face
##        return face.info['rtree'].intersection(envelope)
##    else:
#        # search by traversing the edges of the face
#        edge_ids = set()
#        for signed_edge_id in face.edges:
#            edge_id = positive_id(signed_edge_id)
#            edge = pp.edges[edge_id]
#            if envelope.intersects(edge.geometry.envelope):
#                edge_ids.add(edge_id)
#        return edge_ids

#ALEX: NOT USED?
def check_vertices(pp):
    print("checking vertices")
    ### BEGIN vertex check ###
    # check that all edges in the planar partition
    # have their vertices in the quadtree
    edge_ids_with_missing_vertices = []
    missing_vertices = []
    for edge_id in pp.edges:
        edge = pp.edges[edge_id]
        for pt in edge.geometry:
            if (pt.x, pt.y) not in pp.quadtree:
                edge_ids_with_missing_vertices.append(edge_id)
                missing_vertices.append(pt)
    if edge_ids_with_missing_vertices:
        output_pp_wkt(pp, "step")
        do_output_quadtree(pp)
        print(set(edge_ids_with_missing_vertices))
        for pt in missing_vertices:
            print(pt)
        input("missing vertices in quadtree")
    ### END vertex check ###

#NOTE: I will comment out all the return DB-related operations using #DB, so that I can test without messing with the DB (at first)
def main():
    import time

    start_process_t0 = time.time()

    output = output_layers(OUTPUT_DATASET_NM, SRID)
    #    output = []
    # initialize tables
    with AsyncLoader(workers=1) as loader:
        for table in output:
            loader.load_schema(table)
            del table

    t0 = time.time()
    pp = retrieve(DATASET, SRID, unbounded_id)
    #print(f'Test Alex: {pp}')
    print(f"{time.time()-t0:.3f}s retrieved data from DBMS")

    stepToScale = scalestep.ScaleStep(BASE_DENOMINATOR, DATASET)
    current_denominator = stepToScale.scale_for_step(0)
    print(f"Scale denominator: 1:{current_denominator}")

    # are all vertices in the quadtree ??
    # vertex_check(pp)

    # Dissolve unwanted nodes
    # degree-2
    t0 = time.time()
    dissolve_unwanted_nodes(pp)
    print(f"{time.time()-t0:.3f}s dissolved unwanted nodes (degree 2)")
    # are all vertices in the quadtree ??
    #    vertex_check(pp)

    new_face_id = max(pp.faces.keys()) + 1
    new_edge_id = max(pp.edges.keys()) + 1
    new_node_id = max(pp.nodes.keys()) + 1

    t0 = time.time()
    # faceseq = OrderedSequence(cmp=face_compare)
    faceseq = pqdict.PQDict()  # oid -> priority
    for face in pp.faces.values():
        if face.id != pp.unbounded_id:
            faceseq[face.id] = face.info["area"]  # faceseq.add(face)
    #            faceseq[face.id] = face.info['priority'] # faceseq.add(face)
    print(f"{time.time()-t0:.3f}s priority indexed faces")

    ##    for face in pp.faces.itervalues():
    ##        index_edges_for_face(face, pp)

    t0 = time.time()
    edge_seq = pqdict.PQDict()  # oid -> priority
    for edge in pp.edges.values():
        edge_seq[edge.id] = eps_for_edge_geometry(edge.geometry)
    print(f"{time.time()-t0:.3f}s priority indexed edges")
    face_step = 0

    if do_expensive_post_condition_check_simplify:
        t0 = time.time()
        check_topology_edge_geometry(pp, list(pp.edges.keys()))
        print(f"{time.time()-t0:.3f}s check edge topology (no segments intersecting)")

    #    check_vertices(pp)
    problematic_edge_ids = set([])  # 60329, 61045, 61121]) #16552]) #353, 453]) #[48815])
    problematic_face_ids = set([])  # 4760, 4621, 6136, 7450, 4756, 7360])
    
    if do_edge_simplification:
        # FIXME: de-duplicate code of edge geometry simplification after other
        # generalization operations (split/merge)

        print("starting initial edge generalization")
        t0 = time.time()
        edge_ids_for_simplify = []

        # which scale are we?
        denom_for_step = stepToScale.scale_for_step(face_step)
        # determine line simplify threshold
        step_denom = stepToScale.step_for_scale(denom_for_step)
        ##    small_eps = 0.1
        small_eps = stepToScale.resolution_mpp(denom_for_step)

        # as temporary fix, enter the faces / edge ids here that do
        # cause problems in the process (either they are skipped, or
        # generate debug information)

        # problematic_face_ids = set([5930, 4930])# top10nl_9x9
        # problematic_face_ids = set([4531949])#drenthe

        #    small_eps = 3.
        for _ in range(len(edge_seq)):
            edge_id, eps = edge_seq.topitem()
            if eps < small_eps:
                edge_seq.pop()
                edge_ids_for_simplify.append(edge_id)
            elif eps >= small_eps:
                break
        print(" simplifying {} edges".format(len(edge_ids_for_simplify)))
        for edge_id in edge_ids_for_simplify:

            # FIXME: should we use
            # remove_edge / remove_node here ??
            # add_edge / add_node subsequently ??
            old_edge = pp.edges[edge_id]

            needs_debug = edge_id in problematic_edge_ids
            if needs_debug:
                output_pp_wkt(pp, "step")
                input(f"simplify {edge_id} - starting")
            # simplified_geom, eps = simplify(
            #     old_edge.geometry, pp, tolerance=small_eps, DEBUG=needs_debug
            # )
            simplified_geom, eps = simplify(
                old_edge, pp, small_eps
            )
            new_edge = Edge(
                old_edge.id,
                old_edge.start_node_id,
                angle(simplified_geom[0], simplified_geom[1]),
                old_edge.end_node_id,
                angle(
                    simplified_geom[-1], simplified_geom[-2]
                ),  # angle from last point to second last point
                old_edge.left_face_id,
                old_edge.right_face_id,
                simplified_geom,
                {"step_low": face_step},  # FIXME: should we replace the step-low ??
            )
            pp.edges[edge_id] = new_edge

            #            # check node relationship
            #            for node_id in (new_edge.start_node_id, new_edge.end_node_id):
            #                star = pp.nodes[node_id].star
            #                angles = [get_correct_angle(_, pp.edges) for _ in star]
            #                assert len(set(angles)) == len(angles)

            # if do_expensive_check_after_each_simplification:
            #     """In this situation, we will check the topology after each line is simplified
            #     If it fails, the line is reverted back to its old version
                
            #     NOTE: THIS OPERATION IS VERY EXPENSIVE, FIND ALTERNATIVE TO IT"""
            #     try:
            #         check_topology_edge_geometry(pp, list(pp.edges.keys()))
            #     except:
            #         pp.edges[edge_id] = old_edge


            edge_seq[edge_id] = eps  # _for_edge_geometry(simplified_geom)
            if needs_debug:
                output_pp_wkt(pp, "step")
                input(f"face step - simplify - replaced edge")
        #
        delta = time.time() - t0

        # quadtree
        print(
            " simplified {} edges with small threshold := {:.3f} m^1 ".format(
                len(edge_ids_for_simplify), small_eps
            )
        )
        if do_expensive_post_condition_check_simplify:
            t0 = time.time()
            check_topology_edge_geometry(pp, list(pp.edges.keys()))
            print(
                f"{time.time()-t0:.3f}s check edge topology (no segments intersecting)"
            )
        print(f" {delta:.3f}s simplified edges")

    t0 = time.time()
    processed = 0
    while len(faceseq) > 1:
        ## consistency check (are all points in the quadtree, note -- reversed
        ## -- i.e. all points in quadtree are they still part of an edge? --
        ## is not checked
        ## if face_step % 2500 == 0:
        ##     check_vertices(pp)

        face_step += 1
        face_id = faceseq.pop()  # face.id

        #########################################################
        ## which operation to apply on face? merge or split
        ###
        ## in top10nl feature klass -> split these:
        ##   10xxx == road like object
        ##   12xxx == water like object
        ##
        ## other options, we could consider:
        ## - analyse shape of object here: perimeter versus area (see face.info['ipq'])
        ##   e.g. in top10nl all crossings are tiny squares, might be better to
        ##   merge them instead of split them (leading to many triangular shapes)
        ## - use merge and split in round robin fashion (merge, split, merge, ...)
        ## - it may also be better to split elongated water features
        ## - ...
        op = "merge"
        if pp.faces[face_id].info["feature_class_id"] // 1000 in (10, 12):
            op = "split"
        else:
            op = "merge"

        # we do not apply split, if the face_id is in the set of problematic faces
        if face_id in problematic_face_ids:  # or face_step > 9000:
            op = "merge"
        if do_show_progress:
            print(f"\n#{face_step}")

        #        if face_step == 6863:
        #            output_pp_wkt(pp, "step")

        #        if face_id in (19126,):
        #            output_pp_wkt(pp, f"face_id__{face_id}")

        if op == "merge":
            #########################################################
            ## Merge
            # FIXME: compatibility of faces is not considered
            neighbour_id = find_best_neighbour(face_id, pp)
            if do_show_progress:
                print(
                    " . merging face {} to {} [{}]".format(
                        face_id,
                        neighbour_id,
                        "universe"
                        if neighbour_id == pp.unbounded_id
                        else "non-universe",
                    )
                )
            if neighbour_id is not None:
                face_step, new_face_id, new_edge_id = merge_face_to_neighbour(
                    face_id,
                    neighbour_id,
                    pp,
                    output,
                    faceseq,
                    edge_seq,
                    face_step,
                    new_face_id,
                    new_edge_id,
                )
        elif op == "split":
            #########################################################
            # Split
            #            try:
            denom_for_step = stepToScale.scale_for_step(face_step)
            # determine line simplify threshold
            cur_resolution = stepToScale.resolution_mpp(denom_for_step)
            new_face_id, new_edge_id, new_node_id = split_face(
                face_id,
                pp,
                output,
                faceseq,
                edge_seq,
                face_step,
                new_face_id,
                new_edge_id,
                new_node_id,
                cur_resolution,
            )
        #            except ValueError:
        #                print('fallback to merge')
        #                neighbour_id = find_best_neighbour(face_id, pp)
        #                if neighbour_id is not None:
        #                    face_step, new_face_id, new_edge_id = merge_face_to_neighbour(face_id, neighbour_id, pp, output, faceseq, edge_seq, face_step, new_face_id, new_edge_id)

        ### consistency check post-operation: each face should have at least 1 edge
        #        for tmp_face in pp.faces.itervalues():
        #            assert len(tmp_face.edges) > 0, tmp_face
        #        del tmp_face

        #        if False and op == "split":
        #        if new_face_id == 7097:

        processed += 1
        #########################################################
        ### Find lines that are violating the threshold and simplify them

        if do_edge_simplification:
            # which scale are we?
            denom_for_step = stepToScale.scale_for_step(face_step)
            # determine line simplify threshold
            cur_resolution = stepToScale.resolution_mpp(denom_for_step)
            # cur_resolution *= 0.5  # how many pixels should be 'free' of other vertex
            # check, if we do the reverse (scale to step), we end up at the step
            step_denom = stepToScale.step_for_scale(denom_for_step)

            print(f"denom_for_step: {denom_for_step}, step_denom: {step_denom}, cur_res: {cur_resolution}")
            # simplify lines, store and replace their geometries
            edge_ids_for_simplify = []
            for _ in range(len(edge_seq)):
                edge_id, eps = edge_seq.topitem()
                if eps < cur_resolution:
                    edge_seq.pop()
                    edge_ids_for_simplify.append(edge_id)
                elif eps >= cur_resolution:
                    break
            #        if face_step == 352:
            if not True:
                output_pp_wkt(pp, "step")

            if do_show_progress:  # debugging info about operations
                print(
                    (
                        "   step: {0:d} -> scale: 1:{1:.0f}, "
                        "resolution (m^1): {2:.3f}\n"
                        # "calculated step for scale: {3:10.2f}, "
                        "   #{4} of #{5} edges too detailed".format(
                            face_step,
                            denom_for_step,
                            cur_resolution,
                            step_denom,
                            len(edge_ids_for_simplify),
                            len(pp.edges),
                        )
                    )
                )

            #        print("")
            #        print(cur_resolution, len(edge_ids_for_simplify), edge_ids_for_simplify)
            #        print("")

            # do_edge_simplification

            # FIXME: should we store the old edge before simplification?
            # -------> is an edge simplified over-and-over again?       <-------
            # -------> if so, it is not nice to store all versions...   <-------
            for edge_id in edge_ids_for_simplify:
                #                print(f" {edge_id} will be simplified")
                output_edge(output, pp, edge_id, face_step)

            # -- output some statistics
            # output.edge_stats.append(
            #    *(face_step, len(edge_ids_for_simplify), len(pp.edges))
            # )

            if do_show_progress:
                print("   simplify {} edges".format(len(edge_ids_for_simplify)))
            ##        some = set(list(sorted(edge_ids_for_simplify))[:10])
            ##        print(some)

            for edge_id in edge_ids_for_simplify:
                if do_show_progress:
                    print(f"     simplifying edge {edge_id}")
                old_edge = pp.edges[edge_id]

                ##
                needs_debug = edge_id in problematic_edge_ids
                if needs_debug:
                    output_pp_wkt(pp, "step")
                    input(f"simplify {edge_id} -- {op} - starting")
                ##

                # OLD VERSION: USED FOR VISVALLINGAM_WHYATT
                # simplified_geom, eps = simplify(
                #     old_edge.geometry, pp, tolerance=cur_resolution, DEBUG=needs_debug
                # )
                # = simplified_geom
                # FOR SY Simplification:
                simplified_geom, eps = simplify(
                    old_edge, pp, cur_resolution, DEBUG=needs_debug
                )
                new_edge = Edge(
                    old_edge.id,
                    old_edge.start_node_id,
                    angle(simplified_geom[0], simplified_geom[1]),
                    old_edge.end_node_id,
                    angle(simplified_geom[-1], simplified_geom[-2]),
                    # angle from last point to second last point
                    old_edge.left_face_id,
                    old_edge.right_face_id,
                    simplified_geom,
                    {"step_low": face_step},  # FIXME: should we replace the step-low ??
                )
                pp.edges[edge_id] = new_edge
                edge_seq[edge_id] = eps  # _for_edge_geometry(simplified_geom)
                # FIXME: update stars (incidence to node, given by angle)
                # might have changed, although old value should still be in same
                # part of angle sector around the node
                # FIXME: should we check angle / order of incident edges ???
                if needs_debug:
                    output_pp_wkt(pp, "step")
                    input(f"face step - {op} - simplify - replaced edge")

            if do_expensive_post_condition_check_simplify:
                # expensive post condition check for simplification:
                # no intersections are present in the segments of the edges of the faces
                # having a direct relation with the edges that were simplified
                check_topology_edge_geometry(pp, edge_ids_for_simplify)

        ##            if edge_id in some:
        ##                print(edge_id, eps, len(simplified_geom))

        #        if face_id == 4779:
        #            output_pp_wkt(pp, "step")
        #            input(f"face step - {op} - done")

        #    if ct > 25:
        #        break
        delta = time.time() - t0
        if (processed % 10000) == 0:
            # print delta, "face merging"
            print("{:10d} to be processed \t".format(len(faceseq)), end=" ")
            rate = (60.0 / (delta / processed)) / 60.0
            print("rate per second {0:.0f}".format(rate))
            t0 = time.time()
            processed = 0
            # load data and clear memory
            with AsyncLoader(workers=4) as loader:
                for table in output:
                    loader.load_data(table)
                    table.clear()
                    del table

    #        if rate < 100:
    #            break

    # - finish everything that still remains
    face_step += 1
    # -- faces
    for face_id in pp.faces:
        if face_id != pp.unbounded_id:
            output_face(output, pp, face_id, face_step)
            output_face_hierarchy(output, pp, face_id, pp.unbounded_id, face_step)

    # -- edges
    for edge_id in pp.edges:
        output_edge(output, pp, edge_id, face_step)

    # - load the generated tgap tables to the database
    with AsyncLoader(workers=4) as loader:
        for table in output:
            loader.load_data(table)
            table.clear()
            del table

    # - finalize tables: indexing + clustering
    with AsyncLoader(workers=4) as loader:
        for table in output:
            loader.load_indexes(table)
            del table

    # - finalize tables: statistics
    with AsyncLoader(workers=4) as loader:
        for table in output:
            loader.load_statistics(table)
            del table
    duration_in_secs = time.time() - start_process_t0

    import pprint

    pprint.pprint(STATS_SPLIT_VERTEX_TYPES)
    print(
        "process duration:    {0[0]:d} days {0[1]} hours {0[2]} minutes {0[3]:.3f} seconds".format(
            normalize_seconds(duration_in_secs)
        )
    )


def check_topology_edge_geometry(pp, edge_ids):
    """Check for (self-) intersections of edge geometries"""
    from .edge_simplify.intersection import (
        as_segments,
        segments_intersecting,
        segments_intersection,
    )

    edge_ids = set(edge_ids)
    raised = False
    violations = []
    for edge_id in edge_ids:
        edge = pp.edges[edge_id]
        left_face_id = parent(edge.left_face_id, pp.face_hierarchy)
        right_face_id = parent(edge.right_face_id, pp.face_hierarchy)
        segments_to_check = as_segments(edge.geometry)
        check_edge_ids = set()
        for neighbouring_edge_id in map(positive_id, pp.faces[left_face_id].edges):
            if neighbouring_edge_id != edge_id:
                check_edge_ids.add(neighbouring_edge_id)
        for neighbouring_edge_id in map(positive_id, pp.faces[right_face_id].edges):
            if neighbouring_edge_id != edge_id:
                check_edge_ids.add(neighbouring_edge_id)
        neighbouring_edge_geoms = []
        for check_edge_id in check_edge_ids:
            check_edge = pp.edges[check_edge_id]
            if check_edge.geometry.envelope.intersects(edge.geometry.envelope):
                neighbouring_edge_geoms.append(check_edge.geometry)
        for polyline in neighbouring_edge_geoms:
            segments_to_check.extend(as_segments(polyline))
        try:
            assert not segments_intersecting(segments_to_check)
        except AssertionError:
            raised = True
            print(f"problem with edge {edge_id}")

            pts = segments_intersection(segments_to_check)
            violations.extend(pts)

    if raised:
        output_pp_wkt(pp, "topology_violations")
        with open("/tmp/topology_violations.wkt", "w") as fh:
            fh.write("wkt")
            fh.write("\n")
            for pt in violations:
                fh.write(f"POINT({pt[0]} {pt[1]})")
                fh.write("\n")
        raise ValueError("Topology violated: Edge segments intersect")


def normalize_seconds(seconds):
    """Returns how many days, hours, minutes & seconds there are in a total number of seconds
    """
    (days, remainder) = divmod(seconds, 86400)
    (hours, remainder) = divmod(remainder, 3600)
    (minutes, seconds) = divmod(remainder, 60)
    return (int(days), int(hours), int(minutes), seconds)


def find_best_neighbour(face_id, pp):
    """
    find the best face neighbour to merge with
    
    returns:
        face_id     int
    """
    neighbours = find_neighbours(face_id, pp)
    # find neighbour with longest common boundary
    longest = []
    for neighbour_id in neighbours:
        #        if neighbour_id == pp.unbounded_id:
        #            continue
        cb = common_boundary(face_id, neighbour_id, pp)
        length = sum((pp.edges[edge_id].geometry.length for edge_id in cb))
        tup = (length, neighbour_id)
        longest.append(tup)

    # FIXME: merge with universe?

    # FIXME:
    # could look at
    # * resulting shape measure, would the merge lead to a compact polygon?
    # * compatible faces in the surroundings
    # * continuation of edges / linear features
    neighbour_id = None
    # always increment face step <> (otherwise scale is not determined ok)
    # also, if we do not have a candidate neighbour area for merging with!
    if longest:
        #        print(' ', longest)
        longest.sort(reverse=True)
        _, neighbour_id = longest[0]
        # in case we have another option than the universe, take it
        if neighbour_id == pp.unbounded_id and len(longest) > 1:
            _, neighbour_id = longest[1]
    return neighbour_id


def merge_face_to_neighbour(
    face_id,
    neighbour_id,
    pp,
    output,
    faceseq,
    edge_seq,
    face_step,
    new_face_id,
    new_edge_id,
):
    """
    Merge two faces given by their id
    """
    new_face_id += 1
    if neighbour_id != pp.unbounded_id:
        faceseq.pop(neighbour_id)
        nodes_to_check = merge_face_pair(
            face_id, neighbour_id, new_face_id, pp, edge_seq, output, face_step
        )
    else:
        nodes_to_check = universe_merge_face_pair(
            face_id, neighbour_id, new_face_id, pp, edge_seq, output, face_step
        )

    # merge edges
    pairs = edge_pairs(nodes_to_check, pp)
    for pair, node_id in pairs:
        new_edge_id += 1
        merge_edge_pair(pair, node_id, new_edge_id, pp, edge_seq, output, face_step)

    for node_id in nodes_to_check:
        if node_id in pp.nodes:
            if len(pp.nodes[node_id].star) == 0:
                remove_node(node_id, pp)

    if neighbour_id != pp.unbounded_id:
        # set the new face its priority
        face = pp.faces[new_face_id]
        # materialize geometry for this new face
        # FIXME: if only area and perimeter are needed, we do not need this:
        # we then keep this as thematic/administrative information on a face
        #            wheels = get_wheel_edges(face.edges, pp)
        #            rings = [get_geometry_for_wheel(wheel, pp) for wheel in wheels]
        #            rings = [(abs(ring.signed_area()), ring) for ring in rings]
        #            rings.sort(reverse = True, key=lambda x: x[0])
        #            area, largest_ring = rings[0]
        #            perimeter = largest_ring.length
        iso_perimetric_quotient = (4.0 * math.pi * face.info["area"]) / (
            face.info["perimeter"] * face.info["perimeter"]
        )
        face.info["ipq"] = iso_perimetric_quotient
        # face.info['priority'] = face.info['area'] * face.info['ipq']
        face.info["priority"] = face.info["area"]
        faceseq[new_face_id] = face.info["priority"]
    return face_step, new_face_id, new_edge_id


def split_face(
    face_id,
    pp,
    output,
    faceseq,
    edge_seq,
    face_step,
    new_face_id,
    new_edge_id,
    new_node_id,
    resolution,
):
    #    if face_id == 301:
    #        output_pp_wkt(pp, 'split301')

    #    if face_id in (4781, 4791):
    #        output_pp_wkt(pp, 'step')
    #        input(f'at start of splitting {face_id}')
    if do_show_progress:
        print(f" . splitting face {face_id}")

    #########################################
    # Preparing input for the split operation

    # get all edges that are bounding the face
    edge_ids_in_wheel = set(map(positive_id, pp.faces[face_id].edges))

    assert len(edge_ids_in_wheel) > 0

    # FIXME: We left out classifying a hole for a node as type=2 VertexInfo
    # We should bail out of split and just merge
    # (does not seem to be a point in triangulating and splitting when we just merge?)
    #        neighbours = []
    #        for edge_id in edge_ids_in_wheel:
    #            edge = pp.edges[edge_id]
    #            lf_id = parent(edge.left_face_id, pp.face_hierarchy)
    #            rf_id = parent(edge.right_face_id, pp.face_hierarchy)
    #            neighbour_id = rf_id if face_id == lf_id else lf_id
    #            neighbours.append(neighbour_id)
    #        is_hole = len(set(neighbours)) == 1

    #        wheels = get_wheel_edges(pp.faces[face_id].edges, pp)
    #        neighbours_per_wheel = []
    #        for wheel in wheels:
    #            neighbours = []
    #            for signed_edge_id in wheel:
    #                edge = pp.edges[positive_id(signed_edge_id)]
    #                lf_id = parent(edge.left_face_id, pp.face_hierarchy)
    #                rf_id = parent(edge.right_face_id, pp.face_hierarchy)
    #                neighbour_id = rf_id if face_id == lf_id else lf_id
    #                neighbours.append(neighbour_id)
    #            neighbours_per_wheel.append(neighbours)
    #        print "neighbours_per_wheel", neighbours_per_wheel

    # from these edges get all nodes that are in the boundary of the face
    L = []
    for edge_id in edge_ids_in_wheel:
        edge = pp.edges[edge_id]
        L.extend([edge.start_node_id, edge.end_node_id])
    nodes_incident_to_face = set(L)  # de-dups node id list L

    # FIXME use the stars of the nodes_incident_to_face to get a representation of sectors around a node

    converter = ToPointsAndSegments()
    for node_id in nodes_incident_to_face:
        #        star = pp.stars[node_id]
        star = pp.nodes[node_id].star

        # get node geometry for this node
        edge = pp.edges[positive_id(star[0])]
        g = edge.geometry
        if star[0] < 0:
            v = g[-1]  # in case incoming take last vertex
        else:
            v = g[0]  # in case outgoing take first vertex
        v = (v.x, v.y)
        # END getting the node geometry from the edge
        around = [
            (
                signed_edge_id,
                get_correct_angle(signed_edge_id, pp.edges),
                # get the faces to the left (ccw) and to the right (cw) of the edge as if the edge is pointing out of the node
                get_ccw_face(signed_edge_id, pp.edges, pp.face_hierarchy),
                get_cw_face(
                    signed_edge_id, pp.edges, pp.face_hierarchy
                ),  # << FIXME: not needed?
            )
            for signed_edge_id in star
        ]
        around.sort(key=lambda _: _[1])  # sort on outgoing angle
        # from the info per node, derive how many faces there are around
        # each node
        counts = {}
        for _, _, f, _ in around:
            if f not in counts:
                counts[f] = 1
            else:
                counts[f] += 1

        if face_id not in counts:
            print(counts)
            output_pp_wkt(pp, "split")
            from pprint import pprint

            pprint(around)
            pprint(counts)
            raise ValueError(
                f"node {node_id} does not have a relationship with face {face_id} ?"
            )

        if counts[face_id] > 1:
            # if the face_id is present more than once around a node,
            # this has to be a tangent node!
            tp = 3
            sectors = []
            for one, other in zip(around, around[1:] + [around[0]]):
                e1 = positive_id(one[0])
                beta = one[1]
                e0 = positive_id(other[0])
                alpha = other[1]
                between_face = one[2]
                sector = (
                    between_face,
                    e1,
                    beta,
                    e0,
                    alpha,
                )  # FIXME: change order of e1/e0 to go ccw around node? => should be changed in splitarea as well!
                sectors.append(sector)
        elif len(counts) == 2:
            # we have exactly 2 faces around this node,
            # indicating this is a wheel of edges that forms a hole
            # we have to preserve the face id on the other side
            # of this hole
            tp = 2
            for tmp_face_id in counts:
                if tmp_face_id != face_id:
                    sectors = tmp_face_id
                    break
            else:
                raise ValueError("no other neighbour found")
        else:
            tp = 1
            sectors = None
        info = VertexInfo(tp, sectors, node_id)
        # print(info)
        converter.add_point(v, info=info)

    from splitarea.densify import densify

    densified_geometry = {}
    for edge_id in edge_ids_in_wheel:
        densified_geometry[edge_id] = densify(pp.edges[edge_id].geometry)

    for edge_id in edge_ids_in_wheel:
        edge = pp.edges[edge_id]
        lf_id = parent(edge.left_face_id, pp.face_hierarchy)
        rf_id = parent(edge.right_face_id, pp.face_hierarchy)
        # go over the geometry (but not the end points)
        for pt in densified_geometry[edge_id][1:-1]:  ##edge.geometry[1:-1]:
            v = (pt.x, pt.y)
            tp = 0
            node = None
            sectors = rf_id if face_id == lf_id else lf_id  # take opposite neighbour
            info = VertexInfo(tp, sectors, None)
            # print(info)
            converter.add_point(v, info=info)

    for edge_id in edge_ids_in_wheel:
        edge = pp.edges[edge_id]
        tmp = [(pt.x, pt.y) for pt in densified_geometry[edge_id]]  # edge.geometry]
        for pair in zip(tmp, tmp[1:]):
            converter.add_segment(pair[0], pair[1])

    # from these nodes get all edges that have a relation to the face
    # e.g. also touch the face in only 1 node
    # L = []
    # map(L.extend, [list(map(positive_id, pp.stars[node_id])) for node_id in nodes_incident_to_face])

    L = []
    for node_id in nodes_incident_to_face:
        #        print(f'@node {node_id}')
        #        for signed_id in pp.stars[node_id]:
        for signed_id in pp.nodes[node_id].star:
            L.append(positive_id(signed_id))
    edges_incident_to_face = set(L)
    # by taking all edges
    # minus the composing edges we get the external chains
    # edges_incident_to_face.difference(edge_ids_in_wheel)
    surrounding_edge_ids = edges_incident_to_face - edge_ids_in_wheel

    ext = []
    for edge_id in surrounding_edge_ids:
        edge = pp.edges[edge_id]
        lf_id = parent(edge.left_face_id, pp.face_hierarchy)
        rf_id = parent(edge.right_face_id, pp.face_hierarchy)
        sn_id = edge.start_node_id
        en_id = edge.end_node_id
        ext.append((edge_id, sn_id, en_id, lf_id, rf_id, edge.geometry[:]))
    #    print(ext)

    #########################################
    # Triangulation, while the info is preserved in the triangulation to split

    # -- triangulation / splitting feature
    try:
        assert len(converter.points), "no points given for triangulation"
    except AssertionError:
        raise

    #    try:
    if do_use_grassfire:
        # use grassfire
        from grassfire import calc_skel

        skel = calc_skel(converter, shrink=True, internal_only=True)

        class GrassfireVertexInfo:
            def __init__(self, tp, face_ids, vertex_id):
                self.type = tp
                self.face_ids = face_ids
                self.vertex_id = vertex_id

        class GrassfireTemporaryVertex:
            def __init__(self, pt, info):
                self.x, self.y = pt[0], pt[1]
                self.info = info

        class GrassfireSegments:
            def __init__(self, skel):
                self.skel = skel
                self.segments = []
                self.ext_segments = []
                self._produce_segments()

            def _produce_segments(self):
                """ produce the segments (will be called by constructor) """
                from simplegeom.geometry import Point

                do_output = True
                if do_output:
                    with open("/tmp/nodes_gf.wkt", "w") as fh:
                        fh.write("wkt;info\n")

                for segment in skel.segments():
                    #                    print(segment)
                    (start, end), (start_info, end_info), = segment
                    # see harvest.py in splitarea
                    # TYPE 0
                    # Intermediate vertex on an edge, no need to make connection to this
                    # node
                    #
                    # TYPE 1
                    # node in topology, there needs to be made a connection to this node
                    # => vertex_id is the node_id of the topology
                    #
                    # TYPE 2
                    # E.g. Hole that needs to be dissolved completely, but for that we need
                    # to propagate same label on the whole skeleton!
                    # => face_ids is integer of the face that forms the hole
                    #
                    # TYPE 3
                    # for touching rings, we need to have a node sector list:
                    # angles that bound a certain face, so that we can get the correct face
                    # that overlaps
                    # => face_ids is list with 'node sectors', describing which face is valid for
                    #    which part around the node (based on start and end angle of that sector)
                    if start_info is not None:
                        # top10nl_9x9 (not many touching rings -- really execptional case)
                        # {0: 23595, 1: 19778, 2: 157, 3: 0}
                        #                        if start_info.type == 0:
                        #                            continue

                        STATS_SPLIT_VERTEX_TYPES[start_info.type] += 1
                        v0 = Point(start[0], start[1])
                        v1 = Point(end[0], end[1])
                        lf, rf = None, None
                        if do_output:
                            with open("/tmp/nodes_gf.wkt", "a") as fh:
                                fh.write(f"{v0};{start_info}\n")

                        self.ext_segments.append((v0, v1, lf, rf))
                    else:
                        #                        v0 = Point(start[0], start[1])
                        #                        v1 = Point(end[0], end[1])
                        v0 = GrassfireTemporaryVertex(
                            start, GrassfireVertexInfo(None, None, None)
                        )
                        v1 = GrassfireTemporaryVertex(
                            end, GrassfireVertexInfo(None, None, None)
                        )
                        self.segments.append((v0, v1))

        visitor = GrassfireSegments(skel)
        if True:
            with open("/tmp/skel2.wkt", "w") as fh:
                fh.write("wkt\n")
                for seg in visitor.ext_segments:
                    fh.write(
                        "LINESTRING({0[0].x} {0[0].y}, {0[1].x} {0[1].y})\n".format(seg)
                    )

            with open("/tmp/skel3.wkt", "w") as fh:
                fh.write("wkt\n")
                for seg in visitor.segments:
                    fh.write(
                        "LINESTRING({0[0].x} {0[0].y}, {0[1].x} {0[1].y})\n".format(seg)
                    )
            output_pp_wkt(pp, "split_problem")
            input("paused after grassfire")
    if (
        True
    ):  # FIXME: replace to else: of use_grassfire condition, once making a graph for that works!
        # use splitarea
        try:
            dt = triangulate(converter.points, converter.infos, converter.segments)
        except:
            output_pp_wkt(pp, "split_problem")
            raise

        ###################################
        ##        for edge_id in edge_ids_in_wheel:
        ##            edge = pp.edges[edge_id]
        ##            lf_id = parent(edge.left_face_id, pp.face_hierarchy)
        ##            rf_id = parent(edge.right_face_id, pp.face_hierarchy)
        ##            # go over the geometry
        ##            for pt in edge.geometry[1:-1]:
        ##                v = (pt.x, pt.y)
        ##                tp = 0
        ##                node = None
        ##                sectors = rf_id if face_id == lf_id else lf_id # take opposite neighbour
        ##                converter.add_point(v, info = VertexInfo(tp, sectors, None)

        #        if len(s) == 1:
        #            raise NotImplementedError('hole merge => use merge operation!!!')

        ##        for node_id in nodes_incident_to_face:
        ##            star = pp.stars[node_id]
        ##            signed_edge_id = star[0]
        ##            if signed_edge_id < 0:
        ##                e = pp.edges[~signed_edge_id]
        ##                v = e.geometry[-1]
        ##                v = (v.x, v.y)
        ##            else:
        ##                e = pp.edges[signed_edge_id]
        ##                v = e.geometry[0]
        ##                v = (v.x, v.y)
        ##            tp = 1
        ##            face_ids = None
        ##            node = node_id
        ##            print tp, face_ids, node, "*", v

        #        wheels = get_wheel_edges(pp.faces[face_id].edges, pp)
        #        rings = [get_geometry_for_wheel(wheel, pp) for wheel in wheels]
        #        converter = ToPointsAndSegments()
        #        converter.add_polygon(rings,
        #                              info = VertexInfo(1, 0, None) # type {1|2|3}, face_ids, node_id
        #                                )
        #        ## we could iterate over edges of the face and add just vertices
        #        dt = triangulate_with_infos(converter.points, converter.infos, converter.segments)

        #        try:

        #    it = InteriorTriangleIterator(dt)
        #    interior = [t for t in it]

        it = RegionatedTriangleIterator(dt)
        interior = []
        for group, depth, t in it:
            assert depth in (0, 1, 2)
            if depth == 1:
                interior.append(t)

        if len(interior) == 0:
            raise NotImplementedError(
                "no triangles in the interior of the shape -- fix the line simplification, or enable this workaround - which is not well functioning"
            )
            # no triangles in the interior -- workaround for simplification allowing
            # collapse of edges on top of each other -- to be removed once simplification
            # works fully correct (i.e. fully topologically safe) again
            #
            # let's assume that there are exactly 2 edges running on top of each other
            # bounding this face (stemming from line simplification, simplified the edges too far)
            #
            # note, there should be exactly 2 for our logic to hold
            assert len(pp.faces[face_id].edges) == 2
            # get all edges that are bounding the face
            edge_ids_in_wheel = set(map(positive_id, pp.faces[face_id].edges))
            new_edges = []
            for edge_id in edge_ids_in_wheel:
                edge = pp.edges[edge_id]
                lf_id = parent(edge.left_face_id, pp.face_hierarchy)
                rf_id = parent(edge.right_face_id, pp.face_hierarchy)
                sn_id = edge.start_node_id
                en_id = edge.end_node_id
                new_edge_id += 1
                new_edges.append(
                    [new_edge_id, lf_id, rf_id, sn_id, en_id, edge.geometry]
                )
            # FIXME: we use the first edge of the two, find correct neighbours for these two
            new_edge = new_edges[0]
            other_edge = new_edges[1]
            # figure out which face is on the side that is not the current face
            if other_edge[1] == face_id:
                other_side_face_id = other_edge[2]
            else:
                assert other_edge[2] == face_id
                other_side_face_id = other_edge[1]

            # update the neighbour
            if new_edge[1] == face_id:
                new_edge[1] = other_side_face_id
            else:
                new_edge[2] = other_side_face_id

            # the current face is not one of the neighbours any more
            assert new_edge[1] != face_id
            assert new_edge[2] != face_id

            # remove the old 2 edges
            for edge_id in edge_ids_in_wheel:
                output_edge(output, pp, edge_id, face_step)
                remove_edge(edge_id, pp, edge_seq)

            # add 1 new edge, with correct neighbours
            edge_id, left_face_id, right_face_id, start_node_id, end_node_id, geometry = (
                new_edge
            )
            edge = Edge(
                edge_id,
                start_node_id,
                angle(geometry[0], geometry[1]),
                end_node_id,
                angle(
                    geometry[-1], geometry[-2]
                ),  # angle from last point to second last point
                left_face_id,
                right_face_id,
                geometry,
                {"step_low": face_step},
            )  # FIXME: Keep info on merged edge?
            pp.edges[edge_id] = edge

            # update faces
            pp.faces[parent(left_face_id, pp.face_hierarchy)].edges.add(edge_id)
            pp.faces[parent(right_face_id, pp.face_hierarchy)].edges.add(~edge_id)

            # update nodes
            nodes = pp.nodes
            edges = pp.edges
            edge = pp.edges[edge_id]

            # start node
            if edge.start_node_id not in nodes:
                pt = edge.geometry[0]
                nodes[edge.start_node_id] = Node(edge.start_node_id, pt, [])
                pp.quadtree.add((pt.x, pt.y))
            nodes[edge.start_node_id].star.append(edge_id)
            # end node
            if edge.end_node_id not in nodes:
                pt = edge.geometry[-1]
                pp.quadtree.add((pt.x, pt.y))
                nodes[edge.end_node_id] = Node(edge.end_node_id, pt, [])
            nodes[edges[edge_id].end_node_id].star.append(~edge_id)

            # update quadtree
            first, last = 0, len(edge.geometry) - 1
            for i, pt in enumerate(edge.geometry):
                if i == first or i == last:
                    continue
                pp.quadtree.add((pt.x, pt.y))

            sort_on_angle = partial(get_correct_angle, edges=edges)
            for tmp_node_id in [start_node_id, end_node_id]:
                nodes[node_id].star.sort(key=sort_on_angle)
            del tmp_node_id

            pp.edge_hierarchy[edge_id] = None
            if edge_seq is not None:
                eps = eps_for_edge_geometry(geometry)
                edge_seq[edge_id] = eps

            output_face(output, pp, face_id, face_step)
            # FIXME: output_face_hierarchy ?
            remove_face(face_id, pp)

            return new_face_id, new_edge_id, new_node_id

            # raise ValueError("no triangles (collapsed face for splitting)")
        #    for t in interior:
        #        for v in t.vertices:
        #            print(v.info)
        #        except AttributeError:
        # fhe)
        #            raise
        #        if face_id == 2100:
        #            with open("/tmp/interiortris.wkt", "w") as fh:
        #                output_triangles(interior, fh)

        #    if face_id in (4531949,):
        #        with open("/tmp/interiortris.wkt", "w") as fh:
        #            output_triangles(interior, fh)
        #        input('interior tris ready')

        ########################################
        # Harvest the segments

        # we can choose, which type of skeleton segments
        #            Harvester = MidpointHarvester
        # FIXME: using the midpointharvester Can lead to topology problems,
        # because segments might go outside of the original polygon boundaries,
        # in case there exists a sharp and narrow turn in the polygon
        # (but it does give a nicer, = less jaggy, less sharp angles, center line
        Harvester = EdgeEdgeHarvester
        # Harvester = MidpointHarvester
        # get centerline segments
        visitor = Harvester(interior)
        visitor.skeleton_segments()

        ########################################
        # Connect to nodes
        # make connections to existing edges, in case these are not yet there
        pick = ConnectorPicker(visitor)
        pick.pick_connectors()
    #        with open("/tmp/skel2.wkt", "w") as fh:
    #            fh.write("wkt\n")
    #            for seg in visitor.ext_segments:
    #                fh.write("LINESTRING({0[0].x} {0[0].y}, {0[1].x} {0[1].y})\n".format(seg))

    # -- make graph structure based on segments
    new_edge_id += 1
    new_node_id += 1
    #        try:
    skeleton, new_edge_id = make_graph(
        ext, visitor, new_edge_id, universe_id=unbounded_id, srid=SRID
    )

    if False:  # face_id == 19126:
        with open("/tmp/out_all_raised.wkt", "w") as fh, open(
            "/tmp/out_edges_raised.wkt", "w"
        ) as fhe, open("/tmp/out_it_raised.wkt", "w") as fhit, open(
            "/tmp/out_vertices.wkt", "w"
        ) as fhv:
            #        , open(
            #            "/tmp/ext_edges.wkt", "w"
            #        ) as fhext:
            print("visited", it.visited)
            output_triangles(dt.triangles, fh)
            output_triangles(interior, fhit)
            output_edges(FiniteEdgeIterator(dt, constraints_only=True), fhe)
            output_vertices(dt.vertices, fhv)

        #            # output_external edges
        #            fhext.write("id;sn;sa;lf;rf;wkt\n")
        #            for edge in ext:
        #                fhext.write(";".join(map(str, edge)))
        #                fhext.write("\n")

        #        with open("/tmp/skel2.wkt", "w") as fh:
        #            fh.write("wkt\n")
        #            for seg in visitor.ext_segments:
        #                fh.write(
        #                    "LINESTRING({0[0].x} {0[0].y}, {0[1].x} {0[1].y})\n".format(seg)
        #                )

        #        with open("/tmp/skel3.wkt", "w") as fh:
        #            fh.write("wkt\n")
        #            for seg in visitor.segments:
        #                fh.write(
        #                    "LINESTRING({0[0].x} {0[0].y}, {0[1].x} {0[1].y})\n".format(seg)
        #                )
        input("triangles for split stored")

    label_sides(skeleton)

    for edge in skeleton.half_edges.values():
        assert edge.left_face is not None
        assert edge.right_face is not None

    #        if face_id == 2100:
    #            output_topomap_wkt(skeleton, 'post_label')
    #        raw_input('pause')
    #        output_topomap_wkt(skeleton)
    #        try:
    prune_branches(skeleton)

    #        if face_id == 2100:
    #            output_topomap_wkt(skeleton, 'pruned')
    #        except AssertionError:
    #            output_pp_wkt(pp)
    #            output_topomap_wkt(skeleton, 'fail_assert')
    #            with open('/tmp/out_vertices.wkt', 'w') as fhv:
    #                output_vertices(dt.vertices, fhv)
    #            raise

    ########################################
    # -- make polylines of segments
    groups = define_groups(skeleton)
    new_node_id = make_new_node_ids(skeleton, new_node_id)
    new_edges, new_edge_id = make_new_edges(groups, new_edge_id)

    ########################################
    # -- Remove old edges (all edges with a relation to the face)
    # face_step += 1
    for edge_id in edges_incident_to_face:
        output_edge(output, pp, edge_id, face_step)
        remove_edge(edge_id, pp, edge_seq)

    # -- Add all new edges to the planar partition
    for (
        edge_id,
        start_node_id,
        end_node_id,
        left_face_id,
        right_face_id,
        geometry,
    ) in new_edges:
        # make new edge and
        # add it to the planar partition
        # FIXME: this should be methods add_edge / add_node / add_face
        # in the datastructure module
        pp.edges[edge_id] = Edge(
            edge_id,
            start_node_id,
            angle(geometry[0], geometry[1]),
            end_node_id,
            angle(
                geometry[-1], geometry[-2]
            ),  # angle from last point to second last point
            left_face_id,
            right_face_id,
            geometry,
            {"step_low": face_step},
        )  # FIXME: Keep info on merged edge?
        # update faces
        pp.faces[parent(left_face_id, pp.face_hierarchy)].edges.add(edge_id)
        pp.faces[parent(right_face_id, pp.face_hierarchy)].edges.add(~edge_id)
        # update nodes
        nodes = pp.nodes
        edges = pp.edges
        edge = pp.edges[edge_id]
        # start node
        if edge.start_node_id not in nodes:
            pt = edge.geometry[0]
            nodes[edge.start_node_id] = Node(edge.start_node_id, pt, [])
            pp.quadtree.add((pt.x, pt.y))
        nodes[edge.start_node_id].star.append(edge_id)
        # end node
        if edge.end_node_id not in nodes:
            pt = edge.geometry[-1]
            pp.quadtree.add((pt.x, pt.y))
            nodes[edge.end_node_id] = Node(edge.end_node_id, pt, [])
        nodes[edges[edge_id].end_node_id].star.append(~edge_id)
        # update quadtree
        first, last = 0, len(edge.geometry) - 1
        for i, pt in enumerate(edge.geometry):
            if i == first or i == last:
                continue
            pp.quadtree.add((pt.x, pt.y))

        # sort edges counter clockwise (?) around a node
        sort_on_angle = partial(get_correct_angle, edges=edges)
        for tmp_node_id in [start_node_id, end_node_id]:
            nodes[node_id].star.sort(key=sort_on_angle)
        del tmp_node_id

        pp.edge_hierarchy[edge_id] = None
        if edge_seq is not None:
            eps = eps_for_edge_geometry(geometry)
            edge_seq[edge_id] = eps

    #########################################
    # Output the splittee and remove it
    output_face(output, pp, face_id, face_step)
    # FIXME: output_face_hierarchy
    remove_face(face_id, pp)

    return new_face_id, new_edge_id, new_node_id
    ###################################
    # -- Use the area shares to update the faces around their areas / importances
    ###################################


if __name__ == "__main__":
    main()
