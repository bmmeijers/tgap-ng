from functools import partial
from collections import namedtuple
from simplegeom.geometry import LineString
import math


TAU = math.pi * 2  # https://tauday.com/


# -- record types
Edge = namedtuple(
    "Edge",
    "id, start_node_id, start_angle, end_node_id, end_angle, left_face_id, right_face_id, geometry, info",
)


# class Edge(object):
#    def __init__(
#        self,
#        id,
#        start_node_id,
#        start_angle,
#        end_node_id,
#        end_angle,
#        left_face_id,
#        right_face_id,
#        geometry,
#        info,
#    ):
#        self.id, self.start_node_id, self.start_angle, self.end_node_id, self.end_angle, self.left_face_id, self.right_face_id, self.geometry, self.info = (
#            id,
#            start_node_id,
#            start_angle,
#            end_node_id,
#            end_angle,
#            left_face_id,
#            right_face_id,
#            geometry,
#            info,
#        )
#    def __str__(self):
#        return f"Edge(id={self.id}, {self.start_node_id}, {self.start_angle}, {self.end_node_id}, {self.end_angle}, {self.left_face_id}, {self.right_face_id}, {self.geometry}, {self.info}"


Face = namedtuple("Face", "id, mbr_geometry, pip_geometry, edges, info")
Node = namedtuple("Node", "id, geometry, star")


def positive_id(oid):
    """Return a positive identifier (for an edge)"""
    return oid if oid > 0 else ~oid


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


def get_correct_angle(signed_edge_id, edges):
    if signed_edge_id < 0:
        return edges[~signed_edge_id].end_angle
    else:
        return edges[signed_edge_id].start_angle


def csv_line(lst):
    return ",".join(['"{}"'.format(x) for x in map(str, lst)])


def parent(par, hierarchy):
    """Look up the parent of an object in the hierarchy dict

    We *could* shorten the path in the hierarchy afterwards 
    (with a recursive algorithm this could be on the way out)
    """
    init = last = par
    try:
        while par is not None:
            last = par
            par = hierarchy[par]
    except KeyError:
        raise KeyError("while looking up {} I did not find {}".format(init, par))
    return last


def output_geomgraph_wkt(pp, name=None):
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
        print(
            csv_line(["id", "sn", "sa", "en", "ea", "lf", "rf", "info", "wkt"]), file=fh
        )
        for edge in pp.edges.values():
            coords = ", ".join(
                (f"{pt[0]} {pt[1]}" for pt in edge.geometry)
            )
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
                        edge.info,
                        f"LINESTRING({coords})",
                    ]
                ),
                file=fh,
            )

    with open("/tmp/" + fname, "w") as fh:
        print(csv_line(["id", "wkt", "info"]), file=fh)
        for face in pp.faces.values():
            # "POINT({0[0]} {0[1]})".format(face.pip_geometry)
            #            print >> fh, ";".join(map(str, (face.id, "")))
            if face.pip_geometry is not None:
                geom = "POINT({0[0]} {0[1]})".format(face.pip_geometry)
            else:
                geom = "POINT EMPTY"

            print(csv_line([face.id, geom, face.info]), file=fh)


class GeometryGraph(object):
    """Geometry graph class"""

    def __init__(self, unbounded_id, srid=0):
        self.unbounded_id = unbounded_id
        self.srid = srid

        self.nodes = {}
        self.edges = {}
        self.faces = {}
        self.face_hierarchy = {}
        self.edge_hierarchy = {}

    def add_face(
        self,
        face_id,  # mandatory
        mbr_geometry,
        pip_geometry,
        feature_class_id,  # optional
    ):
        """
        """
        self.faces[face_id] = Face(
            face_id,
            mbr_geometry.envelope
            if mbr_geometry is not None
            else None,  # FIXME store as box2d and map to Envelope at connection level ???
            pip_geometry,
            set([]),
            {"feature_class_id": feature_class_id, "step_low": 0},
        )
        self.face_hierarchy[face_id] = None

    def add_node(self, node_id, geometry):
        self.nodes[node_id] = Node(node_id, geometry, [])

    def add_edge(
        self, edge_id, start_node_id, end_node_id, left_face_id, right_face_id, geometry
    ):
        assert edge_id not in self.edges  # edge should not already be there
        #            pp.edge_rtree.add(edge_id, geometry.envelope)
        self.edges[edge_id] = Edge(
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
        # check for sequential + duplicate points
        for j in range(1, len(geometry)):
            i = j - 1
            assert geometry[i] != geometry[j], geometry
        # add the edge_ids to the edges sets of the faces
        # on the left and right of the edge
        self.faces[left_face_id].edges.add(edge_id)
        self.faces[right_face_id].edges.add(~edge_id)
        # register the edge with the nodes in the stars list
        self.nodes[start_node_id].star.append(edge_id)
        self.nodes[end_node_id].star.append(~edge_id)
        #
        self.edge_hierarchy[edge_id] = None

    def sort_node_stars(self):
        # sort edges counter clockwise (?) around a node
        nodes = self.nodes
        sort_on_angle = partial(get_correct_angle, edges=self.edges)
        for node_id in nodes.keys():
            nodes[node_id].star.sort(key=sort_on_angle)

    # -- modifier functions
    # remove a edge
    def remove_edge(self, edge_id, edge_seq):
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
        edges = self.edges
        face_hierarchy = self.face_hierarchy
        faces = self.faces
        nodes = self.nodes

        # take complement of edge identifier if needed
        edge_id = positive_id(edge_id)
        # remove the edge from the edge dictionary
        edge = edges.pop(edge_id)

        # remove the edge from the edges list of the face
        faces[parent(edge.left_face_id, face_hierarchy)].edges.remove(edge_id)
        faces[parent(edge.right_face_id, face_hierarchy)].edges.remove(~edge_id)
        # remove from node stars list
        nodes[edge.start_node_id].star.remove(edge_id)  # outgoing
        nodes[edge.end_node_id].star.remove(~edge_id)  # incoming

        # FIXME: Remove node when its star list has become 0 length?

        # remove intermediate points from the quadtree
        # (end points of the polyline will be removed when
        # start/end node will be removed)
        if hasattr(self, 'quadtree'):
            first, last = 0, len(edge.geometry) - 1
            for i, pt in enumerate(edge.geometry):
                if i == first or i == last:
                    continue
                pp.quadtree.remove((pt.x, pt.y))
            if edge_seq is not None:
                edge_seq.pop(edge_id)

        return (edge.start_node_id, edge.end_node_id)

    def remove_node(pp, node_id):
        """
        """
        nodes = pp.nodes
        node = nodes.pop(node_id)
        # also remove the point from the quadtree
        if hasattr(pp, 'quadtree'):
            pt = node.geometry
            pp.quadtree.remove((pt.x, pt.y))

    def remove_face(pp, face_id):
        """
        """
        pp.faces.pop(face_id)

    def get_wheel_edges(self, unsorted_edges_for_face):
        edges = self.edges
        nodes = self.nodes
        unsorted_edges = unsorted_edges_for_face.copy()
        edge_id = None
        wheels = []
        while unsorted_edges:
            if edge_id is None:
                wheel = []
                edge_id = unsorted_edges.pop()
            else:
                unsorted_edges.remove(edge_id)
            wheel.append(edge_id)
            print(f"we are at edge#{edge_id} flipped#{~edge_id}")
            if edge_id > 0:
                node_id = edges[edge_id].end_node_id
            else:
                node_id = edges[~edge_id].start_node_id
            star = nodes[node_id].star
            indx = star.index(~edge_id)
            next_edge_id = star[(indx + 1) % len(star)]  # - ccw | + cw
            print(f" next_edge_id = edge#{next_edge_id}")
            if next_edge_id in unsorted_edges:
                edge_id = next_edge_id
                print(f" continuing at edge#{edge_id}")
            else:
                print(" done forming wheel")
                wheels.append(wheel)
                edge_id = None
        return wheels

    def propagate_face_ids_inwards(self, start_edge_ids):
        edges = self.edges
        nodes = self.nodes
        # annotated wheel: (face_id, [signed edge ids])
        annotated_wheels = []
        # for all edges where we need to start
        for signed_edge_id in start_edge_ids:
            #            print(
            #                f"""
            #
            #            """
            #            )
            start_id = signed_edge_id
            wheel = []
            edge = edges[positive_id(signed_edge_id)]
            if signed_edge_id > 0:
                propagate_face_id = edge.right_face_id
            else:
                propagate_face_id = edge.left_face_id
            #            print(f"START at {signed_edge_id} propagating {propagate_face_id}")
            i = 0
            #            while True:
            try:
                for _ in range(len(self.edges) * 2):
                    i += 1

                    #                print(
                    #                    f"""ITERATION {i} at edge#{positive_id(signed_edge_id)} signed:{signed_edge_id}"""
                    #                )
                    wheel.append(signed_edge_id)
                    edge_id = positive_id(signed_edge_id)
                    edge = edges[edge_id]
                    if signed_edge_id > 0:
                        #                    print("rf", edge.right_face_id)
                        if edge.right_face_id is None:
                            #                        edge.right_face_id = propagate_face_id
                            pass
                        #                        print(edge_id, "right needs to be", propagate_face_id)
                        else:
                            assert edge.right_face_id == propagate_face_id
                        node_id = edge.end_node_id
                    else:
                        #                    print("lf", edge.left_face_id)
                        if edge.left_face_id is None:
                            #                        print(edge_id, "left needs to be", propagate_face_id)
                            #                        edge.left_face_id = propagate_face_id
                            pass
                        else:
                            assert edge.left_face_id == propagate_face_id
                        node_id = edge.start_node_id
                    star = nodes[node_id].star
                    indx = star.index(~signed_edge_id)
                    next_edge_id = star[(indx - 1) % len(star)]  # - ccw | + cw
                    signed_edge_id = next_edge_id

                    if next_edge_id == start_id:
                        annotated_wheels.append((propagate_face_id, wheel))
                        wheel = []
                        break
            except AssertionError:
                output_geomgraph_wkt(self, 'assertion_error_propagate')
                raise
        #        print(annotated_wheels)

        #        for face, wheel in annotated_wheels:
        #            print(face, "-->", [idx if idx > 0 else ~idx for idx in wheel])
        #        for edge_id in edges:
        #            print(edges[edge_id])

        #            edge_id = start_edge_id
        #            # form a wheel of edges, until we're back at the start
        #            # a wheel should consist of signed edges (turn or not)
        #            #
        #            # given the wheels and orientation,
        #            # we can propagate the right face onto these edges in a subsequent
        #            # step (or right away)
        #            while not at start:
        #                if edge_id > 0:
        #                    node_id = edges[edge_id].end_node_id
        #                else:
        #                    node_id = edges[~edge_id].start_node_id
        #                star = nodes[node_id].star
        #                indx = star.index(~edge_id)
        #                next_edge_id = star[(indx + 1) % len(star)] # - ccw | + cw
        #                edge_id = next_edge_id
        return annotated_wheels

    def replace_left_right(self, annotated_wheels, debug=False):
#        print(annotated_wheels)

        # invert wheels, i.e. transform the structure:
        #   face_id -> [signed_edge_ids]
        # into a dictionary:
        #   positive edge id -> [left face id, right face id]
        inverted_wheels = {} 
        LEFT = 0
        RIGHT = 1
        for annotated_wheel in annotated_wheels:
            face_id, signed_edge_ids = annotated_wheel
            for signed_edge_id in signed_edge_ids:
                edge_id = positive_id(signed_edge_id)
                if edge_id not in inverted_wheels:
                    inverted_wheels[edge_id] = [None, None]
                if signed_edge_id < 0:
                    idx = RIGHT
                else:
                    idx = LEFT
                inverted_wheels[edge_id][idx] = face_id

        # for all labelled edges, 
        # keep those edges that have 2 different labels on the edge, and both labels need to be not None
        # other edges are removed
        # the final result should be that we have a labelled skeleton of edges on the interior of a face
        for edge_id in sorted(inverted_wheels):
            if None in inverted_wheels[edge_id] or \
                inverted_wheels[edge_id][LEFT] == inverted_wheels[edge_id][RIGHT]:
                self.remove_edge(edge_id, edge_seq = None)
            elif None not in inverted_wheels[edge_id]:
                edge = self.edges[edge_id]
                self.remove_edge(edge_id, edge_seq = None)
                left_face_id, right_face_id = inverted_wheels[edge_id]
                assert left_face_id is not None
                assert right_face_id is not None
                self.add_edge(edge.id,
                              edge.start_node_id, edge.end_node_id,
                              left_face_id, right_face_id,
                              edge.geometry)
        # here we have only the edges that are inside the face and that are properly labelled
        if debug:
            output_geomgraph_wkt(self, "labelled")


    def dissolve_unwanted_nodes(pp):
        """
        Modify planar partition by dissolving edges that share a node
        of degree 2 where this is not needed
        """
        new_edge_id = 1
        if pp.edges:
            new_edge_id += max(pp.edges.keys())

        dissolve = []
        for node_id, node in pp.nodes.items():
            star = node.star
    #    for node_id, star in pp.stars.items():
            if len(star) == 2 and len(set(map(positive_id, star))) == 2:
                pair = tuple(map(positive_id, star))
                dissolve.append((node_id, pair))

        for new_edge_id, (node_id, pair) in enumerate(dissolve, start=new_edge_id):
            pp.merge_edge_pair(pair, node_id, new_edge_id, None, None, 0)

#        for edge_id in pp.edges:
#            print(edge_id, len(pp.edges[edge_id].geometry))

        #output_geomgraph_wkt(pp, "dissolved")


    def merge_edge_pair(pp, pair, middle_node_id, new_edge_id, edge_seq, output, face_step):
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

        for _ in [one, other]:
            assert _.left_face_id is not None, f"edge {_.id} has no left face"
            assert _.right_face_id is not None, f"edge {_.id} has no right face"

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
            pp.remove_edge(tmp_id, edge_seq = None)
        del tmp_id
        #
        pp.remove_node(middle_node_id)
        assert left_face_id is not None
        assert right_face_id is not None
        pp.add_edge(new_edge_id, 
            start_node_id,
            end_node_id,
            left_face_id,
            right_face_id,
            geometry,
        )

        edge_hierarchy[one_id] = new_edge_id
        edge_hierarchy[other_id] = new_edge_id

        #    vertex_check(pp)
        # make new edge
#        edge_id = new_edge_id
#        # add it to the planar partition
#        edges[edge_id] = Edge(
#            edge_id,
#            start_node_id,
#            angle(geometry[0], geometry[1]),
#            end_node_id,
#            angle(geometry[-1], geometry[-2]),  # angle from last point to second last point
#            left_face_id,
#            right_face_id,
#            geometry,
#            {"step_low": face_step},
#        )  # FIXME: Keep info on merged edge?

#        # DO_SIMPLIFY
#        if hasattr(pp, 'quadtree'):
#            for pt in geometry:
#                #        print('adding;POINT({0} {1})'.format(pt.x, pt.y ) )
#                pp.quadtree.add((pt.x, pt.y))
#            #        is_found = pp.kdtree.undelete([pt.x ,pt.y]) # FIXME: first/last point different?
#            #        assert is_found
#        #    vertex_check(pp)
#        # update faces
#        faces[parent(left_face_id, face_hierarchy)].edges.add(edge_id)
#        faces[parent(right_face_id, face_hierarchy)].edges.add(~edge_id)
#        # update stars
#    #    stars[edges[edge_id].start_node_id].append(edge_id)  # outgoing: +
#    #    stars[edges[edge_id].end_node_id].append(~edge_id)  # incoming: -
#        nodes[edges[edge_id].start_node_id].star.append(edge_id)  # outgoing: +
#        nodes[edges[edge_id].end_node_id].star.append(~edge_id)  # incoming: -

#        sort_on_angle = partial(get_correct_angle, edges=edges)
#        for tmp_id in [start_node_id, end_node_id]:
#    #        stars[tmp_id].sort(key=sort_on_angle)
#            nodes[tmp_id].star.sort(key=sort_on_angle)
#        del tmp_id
#        # add merge to edge_hierarchy


#        # DO_SIMPLIFY
#        # print('made new edge', edge_id)
#        if edge_seq is not None:
#            eps = eps_for_edge_geometry(geometry)
#            edge_seq[edge_id] = eps
    def relabel_nodes(self, nodes_mapping, new_node_id, new_edge_id):
        edges_new = []
        for edge_id in self.edges:
            edge = self.edges[edge_id]
            if edge.start_node_id in nodes_mapping:
                start_node_id = nodes_mapping[edge.start_node_id]
            else:
                new_node_id += 1
                nodes_mapping[edge.start_node_id] = new_node_id
                start_node_id = new_node_id

            if edge.end_node_id in nodes_mapping:
                end_node_id = nodes_mapping[edge.end_node_id]
            else:
                new_node_id += 1
                nodes_mapping[edge.end_node_id] = new_node_id
                end_node_id = new_node_id
            new_edge_id += 1
            edges_new.append((new_edge_id, start_node_id, end_node_id, edge.left_face_id, edge.right_face_id, LineString(edge.geometry, srid=self.srid)))

#        for edge_new in edges_new:
#            print(edge_new)

#        with open("/tmp/final_edges.csv", "w") as fh:
#            print(
#                csv_line(["id", "sn", "en", "lf", "rf", "info", "wkt"]), file=fh
#            )
#            for edge_new in edges_new:
#                coords = ", ".join(
#                    (f"{pt[0]} {pt[1]}" for pt in edge_new[-1])
#                )
#                print(
#                    csv_line(
#                        [
#                            edge_new[0],
#                            edge_new[1],
#                            edge_new[2],
#                            edge_new[3],
#                            edge_new[4],
#                            None,
#                            f"LINESTRING({coords})",
#                        ]
#                    ),
#                    file=fh,
#                )


        return edges_new, new_node_id, new_edge_id

# algo should create from unlabelled geometric segments
# labelled segments
# SRID!
def label_segments(labelled, unlabelled, splittee_id, new_node_id, new_edge_id, unbounded_id=0, srid=0):
    nodes_mapping = {}
    graph = GeometryGraph(unbounded_id=unbounded_id, srid=srid)
    labelled_edge_ids = []
    for (
        edge_id,
        start_node_id,
        end_node_id,
        left_face_id,
        right_face_id,
        geometry,
    ) in labelled:
        # we ignore the start_node_id/end_node_id
        # we should keep a dictionary here, so we can map the node geometry back to node_id

        start_node = geometry[0]
        end_node = geometry[-1]

        # dictionary node geometry to node id
        if start_node_id is not None:
            nodes_mapping[start_node] = start_node_id
        if end_node_id is not None:
            nodes_mapping[end_node] = end_node_id

        if left_face_id not in graph.faces:
            graph.add_face(left_face_id, None, None, None)
        if right_face_id not in graph.faces:
            graph.add_face(right_face_id, None, None, None)

        if start_node not in graph.nodes:
            graph.add_node(start_node, start_node)
        if end_node not in graph.nodes:
            graph.add_node(end_node, end_node)

        graph.add_edge(
            edge_id, start_node, end_node, left_face_id, right_face_id, geometry
        )
        labelled_edge_ids.append(edge_id)
    graph.add_face(None, None, None, None)

    for geometry in unlabelled:
        start_node, end_node = geometry[0], geometry[-1]
        edge_id = max(graph.edges.keys()) + 1  # FIXME: should be globally unique

        if start_node not in graph.nodes:
            graph.add_node(start_node, start_node)
        if end_node not in graph.nodes:
            graph.add_node(end_node, end_node)

        graph.add_edge(edge_id, start_node, end_node, None, None, geometry)

    # IMPORTANT, sort the edges around the nodes
    graph.sort_node_stars()

#    for elements in [graph.nodes, graph.edges, graph.faces]:
#        print("")
#        for element in elements.values():
#            print("*", repr(element))

    start_edges = []
    for edge in graph.edges.values():
        # skip edges without neighbour (these we want to label, not start from)
        if edge.left_face_id is None or edge.right_face_id is None:
            continue

        if edge.left_face_id == splittee_id:
            assert edge.right_face_id != splittee_id
            start_edges.append(+edge.id)
            # propagate edge.right_face_id
        else:
            assert edge.right_face_id == splittee_id
            start_edges.append(~edge.id)
            # propagate edge.left_face_id

#    print(start_edges)

    # to_visit = set(graph.edges.keys())
    # print(to_visit)

    # tmp = []
    # for edge_id in labelled_edge_ids:
    #    tmp.append(~edge_id)
    # labelled_edge_ids.extend(tmp)
    # to_visit = set(labelled_edge_ids)
    # print(graph.get_wheel_edges(to_visit))

    ann_wheels = graph.propagate_face_ids_inwards(start_edges)

    # if splittee_id in (20384,):
    #     output_geomgraph_wkt(graph, 'propagated')

    # inverts annotated wheels into left/right info
    graph.replace_left_right(ann_wheels) #, debug=splittee_id in (20384,))

    for edge in graph.edges.values():
        assert edge.left_face_id is not None
        assert edge.right_face_id is not None

    graph.dissolve_unwanted_nodes()

    #output_geomgraph_wkt(graph, "after")

    # output all relevant edges that have been preserved
    # output_geomgraph_wkt(graph, 'selection')

    # new_node_id = 1000000
    final_edges, new_node_id, new_edge_id = graph.relabel_nodes(nodes_mapping, new_node_id, new_edge_id)
    return final_edges, new_node_id, new_edge_id


if __name__ == "__main__":
    pass
