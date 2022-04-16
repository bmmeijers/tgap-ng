from __future__ import annotations #used for type-hinting for lists (like list[Class])
from enum import IntEnum
import traceback
from tgap_ng.datastructure import PlanarPartition, Edge, eps_for_edge_geometry

from shapely import wkt, errors as shpErr
from shapely.geometry import Point as shpPoint, LineString as shpLS
from simplegeom import geometry as simplgeom

import matplotlib.pyplot as plt

from .SY_utils import plotShpLS, convertSimplPtsToShp
from .SY_DataStrucuctures import SegmentCollention, ObjectNotCreatedException
from .SY_constants import TopologyIssuesException, PreClassificationException

from math import sqrt, pow

def returnIndexesOfSearchedElems(listToBeSearched: list, queryKeywords: list) -> dict:
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

def simplifySYSimple(edgeToBeSimplified: Edge, pp: PlanarPartition, tolerance, DEBUG = False, showPlots = False):
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

    try:
        geom = wkt.loads(edgeToBeSimplified.geometry.wkt)
    except shpErr.WKTReadingError as err:
        print(f"Error while transforming the geom.wkt to shp LineString: {err}")

    #plot the initial geometry
    if showPlots:
        plotShpLS(geom, "red")

    ptsList = list(geom.coords)

    # Create a point list containing the Shapely Points, and from that generate a segment List
    shpPtList = convertSimplPtsToShp(ptsList)
    
    # TODO create a SegmentCollection Object
    try:
        segColl = SegmentCollention(shpPtList, pp, edgeToBeSimplified)
    except ObjectNotCreatedException:        
        print("The Segment Collection could not be created")
        # TODO: WHAT SHOULD I DO ONE IT FAILS?
        return edgeToBeSimplified.geometry, eps_for_edge_geometry(edgeToBeSimplified.geometry)

    try:
        newEdge = segColl.simplify(shpPtList, ptsList)
        return newEdge.geometry, eps_for_edge_geometry(newEdge.geometry)
    except TopologyIssuesException:
        print("Returning original edge")
        return edgeToBeSimplified.geometry, eps_for_edge_geometry(edgeToBeSimplified.geometry)
    except PreClassificationException:
        print("COULD NOT PERFORM A CORRECT CLASSIFICATION. Returning original edge")
        return edgeToBeSimplified.geometry, eps_for_edge_geometry(edgeToBeSimplified.geometry)
    except Exception as e:
        print (f"Random Exception: {e}, {traceback.format_exc()}")
        return edgeToBeSimplified.geometry, eps_for_edge_geometry(edgeToBeSimplified.geometry)
    

    print("Test print")
    
    
