from random import random, seed
from math import sqrt, cos, sin, pi

from tgap_ng.datastructure import parent, PlanarPartition, Edge

from shapely import wkt
from shapely.geometry import LineString as shpLS, Point as shpPT

from tgap_ng.edge_simplify.SY_utils import plotShpLS
from tgap_ng.edge_simplify.SY_DataStrucuctures import convertPtListToSimplGeomLS
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

def checkIntersectionSimplifiedSegWithNeighbouringSegments(originalEdge: Edge, simplifiedEdge: shpLS,
        pp: PlanarPartition, nodesToIgnore: list) -> bool:
    leftFaceId = parent(originalEdge.left_face_id, pp.face_hierarchy)
    rightFaceId = parent(originalEdge.right_face_id, pp.face_hierarchy)

    leftNeighEdges = list(pp.faces[leftFaceId].edges)
    rightNeighEdges = list(pp.faces[rightFaceId].edges)

    #modify the list of segments to segments by turning them positive + removing the initial edge + removing doubles
    # and don't check with global face (?? Will it create Topo problems?)
    # commented because I want to get all edge, even the ones from the "world" face
    # if rightFaceId == 0:
    #     allNeigh = leftNeighEdges
    # elif leftFaceId == 0:
    #     allNeigh = rightNeighEdges
    # else:
    allNeigh = leftNeighEdges + rightNeighEdges
    for neighId in range(0,len(allNeigh)):
        if allNeigh[neighId] < 0:
            oldVal = allNeigh[neighId]
            allNeigh[neighId] = ~oldVal

    #allNeighPositive = list(map(abs, allNeigh))
    allNeighPosUnique = list(dict.fromkeys(allNeigh))

    for edgeId in allNeighPosUnique:
        if edgeId == originalEdge.id:
            # We may come across our own edge before simplification, skip it
            continue
        
        edgeGeom = pp.edges[edgeId].geometry

        shpEdge = wkt.loads(edgeGeom.wkt)
        
        intersObj = simplifiedEdge.intersection(shpEdge)

        #TODO: Fix this part!
        if type(intersObj) is shpPT and intersObj.coords[0] not in nodesToIgnore:
            return False            
        if type(intersObj) is not shpLS and type(intersObj) is not shpPT:
            #print("The intersection does NOT result in a simple shape, thus no need to look further")
            #plotShpLS(intersObj, "red")
            return False

        # No need to work with other data types such as GeometryCollection
        intersList = [*intersObj.coords] # solution from: github.com/shapely/issues/630
        #plotShpLS(intersObj, "red")
        if len(intersList) != 0:
            # in this situation, we do have an interection between our simplified line and one of its neighbours. 
            # We should now see if this intersection point is one of O

            if len(intersList) > 1:
                # we intersect in two or more points, that is not ok.
                

                intersPt = intersList[0] #now we know we only have one intersection coordinate, check for that

                if intersPt not in nodesToIgnore:
                    return False
            
    return True

def convertShpLSToSimpleGeom(shpInstance: shpLS):
    """Used to convert from a Shapely Line String to a simple geometry
    First creates a list of points, then converts that to an Edge geometry 
    """
    coords = shpInstance.coords
    ptList = []
    for x, y in coords:
        ptList.append((x,y))

    new_geom = convertPtListToSimplGeomLS(ptList)
    return new_geom

def shapelyIntersection(originalEdge: Edge, pp: PlanarPartition):
    shpEdgeGeom = wkt.loads(originalEdge.geometry.wkt)

    startNodeGeom = pp.nodes[originalEdge.start_node_id].geometry
    endNodeGeom = pp.nodes[originalEdge.end_node_id].geometry

    startPoint = (startNodeGeom.x, startNodeGeom.y)
    endPoint = (endNodeGeom.x, endNodeGeom.y)

    mainNodesList = [startPoint, endPoint]

    simplified_geom_shp = shpEdgeGeom.simplify(0.3)
    #plotShpLS(shpEdgeGeom, "red")
    #plotShpLS(simplified_geom_shp, "green")

    if checkIntersectionSimplifiedSegWithNeighbouringSegments(originalEdge,simplified_geom_shp, pp, mainNodesList) is False:
        return originalEdge.geometry
    else:
        #print(f"Simplification completed successfully using Shapely for is{originalEdge.id}")
        return convertShpLSToSimpleGeom(simplified_geom_shp)