from __future__ import annotations #used for type-hinting for lists (like list[Class])
from tgap_ng.datastructure import PlanarPartition, Edge, eps_for_edge_geometry, parent

from shapely import wkt, errors as shpErr
from shapely.geometry import Point as shpPoint, LineString as shpLS
from simplegeom import geometry as simplgeom

import matplotlib.pyplot as plt

from .SY_utils import plotShpLS, convertSimplPtsToShp, decideBias, reverseBias
from .SY_DataStrucuctures import SegmentCollention, ObjectNotCreatedException
from .SY_constants import TopologyIssuesException, PreClassificationException

from math import sqrt, pow

from .utils import checkIntersectionSimplifiedSegWithNeighbouringSegments

def returnIndexesOfSearchedElems(listToBeSearched: list, queryKeywords: list) -> dict:
    # NOT USED???
    # Returns the indexes which contain one of the keywords in the queries list
    # in the form a dictionary key=enum -> value=list[idx]
    resultDict = {}
    for qK in queryKeywords:
        resultDict[qK] = []

    for i in range(0, len(listToBeSearched)):
        elem = listToBeSearched[i]
        if elem in queryKeywords:
            # add the index associated to that element
            resultDict[elem].append(i)

    return resultDict

def simplifySYSimple(edgeToBeSimplified: Edge, pp: PlanarPartition, tolerance, DEBUG = False, showPlots = False, count_success = [0,0,0,0,0]):
    """
    Method used for simplifying a polyline having characteristics of man-made structures
    (i.e. orthogonal turns inbetween segments)

    Simplification based on the paper "Shape-Adaptive Geometric Simplification of Heterogeneous Line Datasets"
    by Samsonov and Yakimova. Further info on this algorithm can be found in 3.1.1. Simpl. of orthogonal segments.

    Topological awareness also needs to be accounted for, both in the case of self-intersections (which are mentioned
    in the paper) as well as with neighbouring structures (which is beyond the scope of that paper).
    """
    #print("Entered SY Simplification Module - v2.1 - Shortcut and Diagonal only")
    #print(f"Original Edge: {edgeToBeSimplified.geometry}")

    startNodeGeom = pp.nodes[edgeToBeSimplified.start_node_id].geometry
    endNodeGeom = pp.nodes[edgeToBeSimplified.end_node_id].geometry

    startPoint = (startNodeGeom.x, startNodeGeom.y)
    endPoint = (endNodeGeom.x, endNodeGeom.y)

    mainNodesList = [startPoint, endPoint]

    #print(f"Starting the SY Simplification for edge {edgeToBeSimplified.id}")
    try:
        geom = wkt.loads(edgeToBeSimplified.geometry.wkt)
    except shpErr.WKTReadingError as err:
        print(f"Error while transforming the geom.wkt to shp LineString: {err}")

    #plot the initial geometry
    if showPlots:
        plotShpLS(geom, "red")

    ptsList = list(geom.coords)
    #print(f"Initial no of points {len(ptsList)}")
    # Create a point list containing the Shapely Points, and from that generate a segment List
    shpPtList = convertSimplPtsToShp(ptsList)
    
    # TODO create a SegmentCollection Object
    try:
        segColl = SegmentCollention(shpPtList, pp, edgeToBeSimplified)
    except ObjectNotCreatedException:        
        #print("The Segment Collection could not be created")
        # TODO: WHAT SHOULD I DO ONE IT FAILS?
        count_success[1] += 1
        return edgeToBeSimplified.geometry, eps_for_edge_geometry(edgeToBeSimplified.geometry)

    try:
        bias = decideBias()
        newgeom: shpLS = segColl.simplify(shpPtList, ptsList, bias)
    #except TopologyIssuesException:\
        #NOT EVEN USED!!!!
        # print("SYERROR - Returning original edge")
        #segColl.trxManager.rollback(pp.quadtree)
        #count_success[1] += 1
        #return edgeToBeSimplified.geometry, eps_for_edge_geometry(edgeToBeSimplified.geometry)
    except PreClassificationException:
        #print("SYERROR - COULD NOT PERFORM A CORRECT CLASSIFICATION. Returning original edge")
        #segColl.trxManager.rollback(pp.quadtree)
        count_success[2] += 1
        return edgeToBeSimplified.geometry, eps_for_edge_geometry(edgeToBeSimplified.geometry)
    except Exception as e:
        #print (f"Random Exception: {e}, {traceback.format_exc()}. Retruning original edge")
        #segColl.trxManager.rollback(pp.quadtree)
        count_success[2] += 1
        return edgeToBeSimplified.geometry, eps_for_edge_geometry(edgeToBeSimplified.geometry)

    # plot the initial geometry
    if showPlots:
        plotShpLS(newgeom, "green")

    ##################################################
    # Now that we have a simplified edge, we have to perform a topological check.
    # Note: The solution proposed below doesn't seem to be the most efficient, but HOPEFULLY it will work
    #
    # We know that, since the planar partition adheres initially (i.e. b4 simplification) to topological consistencies rules,
    # after the simplification, we can check the intersections between the new LS and all other geometries having either left or right face in common
    # extract those geometries from PP, transform them into Shapely LS, and then perform the .intersects() opertaion

    # First, check for self-interections. That is also a topological error, and should be handled accordingly
    if not newgeom.is_simple:
        #print(f"Our simplified line from edge {edgeToBeSimplified.id} results in a SELF-INTERSECTION. Retruning original edge")
        #segColl.trxManager.rollback(pp.quadtree)
        # RESERVE BIAS CODE, BUT DOES NOT GIVE BETTER RESULT!!!
        #segColl2 = SegmentCollention(shpPtList, pp, edgeToBeSimplified)
        #try:
        #    newgeom2: shpLS = segColl2.simplify(shpPtList, ptsList, reverseBias(bias))
        #except:
        count_success[3] += 1
        return edgeToBeSimplified.geometry, eps_for_edge_geometry(edgeToBeSimplified.geometry)
        #if not newgeom2.is_simple:
        #    count_success[3] += 1
        #    return edgeToBeSimplified.geometry, eps_for_edge_geometry(edgeToBeSimplified.geometry)
        #newgeom = newgeom2

    if checkIntersectionSimplifiedSegWithNeighbouringSegments(edgeToBeSimplified, newgeom, pp, mainNodesList) is False:
        #segColl.trxManager.rollback(pp.quadtree)
        count_success[4] += 1
        return edgeToBeSimplified.geometry, eps_for_edge_geometry(edgeToBeSimplified.geometry)
    
    #print(f"SIMPLIFICATION PERFORMED SUCCESSFULLY! Edge with id {edgeToBeSimplified.id} has been successfully simplified, and no topological issues have been detected")

    segColl.trxManager.commit(pp.quadtree)
    finalEdge = segColl.returnFinalEdge(ptsList)

    count_success[0] += 1

    return finalEdge, eps_for_edge_geometry(finalEdge)

    

