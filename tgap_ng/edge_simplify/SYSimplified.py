from enum import IntEnum
from tgap_ng.datastructure import PlanarPartition, Edge

from shapely import wkt, errors as shpErr
from shapely.geometry import Point as shpPoint, LineString as shpLS
from simplegeom import geometry as simplgeom

import matplotlib.pyplot as plt

from .SY_utils import plotShpLS, convertSimplPtsToShp
from .SY_DataStrucuctures import SegmentCollention

from math import sqrt, pow

from __future__ import annotations #used for type-hinting for lists (like list[Class])

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

def simplifySYSimple(edgeToBeSimplified: Edge, pp: PlanarPartition, tolerance, DEBUG = False):
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
    plotShpLS(geom, "red")

    ptsList = list(geom.coords)

    # Create a point list containing the Shapely Points, and from that generate a segment List
    shpPtList = convertSimplPtsToShp(ptsList)
    
    # TODO create a SegmentCollection Object
    segColl = SegmentCollention(shpPtList)

    if segColl is None:
        print("The Segment Collection could not be created")
        # TODO: WHAT SHOULD I DO ONE IT FAILS?
        return

    
    # # Check that the configuration is correct
    # # It should contain either two Segment Extensions (Start/End) + Replace or One Point Extension + Segment Extension
    # keywrds = [Operations.SEG_EXTEND_START, Operations.SEG_EXTEND_END, Operations.PT_EXTEND_START, 
    #            Operations.PT_EXTEND_END, Operations.REPLACE]
    # keywrdsSearchIdxs: dict = returnIndexesOfSearchedElems(operToApplyToSegList, keywrds)

    # # [SegExtStartList, SegExtEndList, PtExtStartList, PtExtEndList, SegGenReplaceList]
    # simplConfigOper = [keywrdsSearchIdxs[Operations.SEG_EXTEND_START], keywrdsSearchIdxs[Operations.SEG_EXTEND_END],
    #                    keywrdsSearchIdxs[Operations.PT_EXTEND_START], keywrdsSearchIdxs[Operations.PT_EXTEND_END],
    #                    keywrdsSearchIdxs[Operations.REPLACE]]

    # segExtStartOperNo = len(simplConfigOper[0]) # SegExtStartList
    # segExtEndOperNo = len(simplConfigOper[1]) # SegExtEndList
    # ptExtStartOperNo = len(simplConfigOper[2]) # PtExtStartList
    # ptExtEndOperNo = len(simplConfigOper[3]) # PtExtEndList
    # replaceOperNo = len(simplConfigOper[4]) # SegGenReplaceList

    # simplConfigNo = [segExtStartOperNo, segExtEndOperNo, ptExtStartOperNo, ptExtEndOperNo, replaceOperNo]


    # # Acceptable configurations: 1 Segment_Start, 1 Segment_End, 1 Replace (for Circular: only this is available!! - as we don't have any end points)
    # # 1 Segment Start + 1 Point Start, no Replace OR 1 Segment End + 1 Point End, no Replace
    # extendLeftRightNeighFlag = False
    # ExtendNeighToPointStartFlag = False
    # ExtendNeighToPointEndFlag = False

    # if simplConfigNo == [1,1,0,0,1]:
    #     print("We have to extend the 2nd degree neighbours (left/right) and replace the shortest segment with its perpendicular")
    #     extendLeftRightNeighFlag = True
    # elif simplConfigNo == [1,0,1,0,0]:
    #     print("Determine the perpendicular from point to extension of segment - START")
    #     ExtendNeighToPointStartFlag = True
    # elif simplConfigNo == [0,1,0,1,0]:
    #     print("Determine the perpendicular from point to extension of segment - END")
    #     ExtendNeighToPointEndFlag = True
    # else:
    #     print("SOMETHING WENT WRONG WITH CREATING A SIMPLIFICATION STRUCTURE")
    #     raise Exception

    # simplifyMedian = False # choose wheter to go with the median or shortcut simplification

    # if extendLeftRightNeighFlag:
    #     # in this case, we will replace our shortest segment with its perpendicular
    #     # determine its start by extending the segment_END to this new line
    #     # detemine its end by extending the segment_START to it

    #     # TODO: add checks to ensure that there are no self-intersections

    #     extendSegStartIdx = simplConfigOper[0][0] # First (and only) id in the SegExtStartList
    #     extendSegEndIdx = simplConfigOper[1][0]
    #     replSegIdx = simplConfigOper[4][0]

    #     extendSegStart: Segment = segList[extendSegStartIdx]
    #     extendSegEnd: Segment = segList[extendSegEndIdx]
    #     replSeg: Segment = segList[replSegIdx]

    #     #retrieve coordinates

    #     # coords of the segment to be extended from the START
    #     extdStartSegP1: shpPoint = shpPtList[extendSegStart.startId]
    #     extdStartSegP2: shpPoint = shpPtList[extendSegStart.endId]

    #     # coords of the segment to be extended from the END
    #     extdEndSegP1: shpPoint = shpPtList[extendSegEnd.startId]
    #     extdEndSegP2: shpPoint = shpPtList[extendSegEnd.endId]

    #     # coords of the segment to be REPLACED
    #     replSegP1: shpPoint = shpPtList[replSeg.startId]
    #     replSegP2: shpPoint = shpPtList[replSeg.endId]

    #     if simplifyMedian:
    #         # MEDIAN SIMPLIFICATION
    #         # we've already checked that in this list there is only one element (id), so we want to extract those

    #         # Determine the line equation of extdStartSeg 
    #         extdStartSegLineEq = LineEquation(extdStartSegP1, extdStartSegP2)

    #         # Determine the line equation of extdEndSeg 
    #         extdEndSegLineEq = LineEquation(extdEndSegP1, extdEndSegP2)

    #         # REPALCE THE SHORTEST SEGMENT
    #         # Determine the slope of replSeg 
    #         replSeg_slope = (replSegP2.y - replSegP1.y)/(replSegP2.x - replSegP1.x)

    #         # determine the MIDPOINT of the segment to be replaced
    #         midPoint = shpPoint((replSegP1.x+replSegP2.x)/2,(replSegP1.y+replSegP2.y)/2)

    #         # the slope of the line PERPendicular to our replaced line is -(1/m)
    #         perpSeg_slope = -1/replSeg_slope
    #         perpSeg_yintercept = midPoint.y - perpSeg_slope*midPoint.x

    #         perpSegLineEq = LineEquation(perpSeg_slope, perpSeg_yintercept)

    #         # Intersect extdStartSeg & perpSeg to determine the intersection point
    #         newPoint_extdStartSeg = intersectionPoint(extdStartSegLineEq, perpSegLineEq)
    #         newPoint_extdEndSeg = intersectionPoint(perpSegLineEq, extdEndSegLineEq)

    #         newSegList = []

    #         ptsList.append(newPoint_extdStartSeg)
    #         newPoint_extdStartIdx = len(ptsList) - 1 #last one added to the list

    #         ptsList.append(newPoint_extdEndSeg)
    #         newPoint_extdEndIdx = len(ptsList) - 1 #last one added

    #         # TODO: Check if Median Substitution returns correct result, else use Point to Line subst
    #         # replSegExtStart = 

    #         for operToApplyIdx in range(0,len(operToApplyToSegList)):
    #             operToApply = operToApplyToSegList[operToApplyIdx]
    #             # if it is Operations.REMOVE do nothing
    #             if operToApply is Operations.KEEP:
    #                 newSegList.append(segList[operToApplyIdx])
    #             elif operToApply is Operations.REPLACE:
    #                 replacementSegment = Segment(newPoint_extdEndIdx, newPoint_extdStartIdx)
    #                 newSegList.append(replacementSegment)
    #             elif operToApply is Operations.SEG_EXTEND_START:
    #                 oldSegment: Segment = segList[operToApplyIdx]
    #                 newSegment = Segment(newPoint_extdStartIdx, oldSegment.endId)
    #                 newSegList.append(newSegment)
    #             elif operToApply is Operations.SEG_EXTEND_END:
    #                 oldSegment: Segment = segList[operToApplyIdx]
    #                 newSegment = Segment(oldSegment.startId, newPoint_extdEndIdx)
    #                 newSegList.append(newSegment)

    #         newEdge = convertSegListToSimplGeomLS(newSegList, ptsList, edgeToBeSimplified)

    #         # test to see if geometry is simple
    #         newGeomShp: shpLS = wkt.loads(newEdge.geometry)
    #         if not newGeomShp.is_simple:
    #             # Sometimes, it could happen that a Median intersection could result in a wrong result
    #             # Should that happen, we can use a "point perp on line" intersection instead
    #             print("Something went wrong!")
    #         return newEdge
    #     else:
    #         # we change the segment which was supposed to have its ending extended with Pt_ext_end
    #         extdSegLineEq: LineEquation = LineEquation(extdStartSegP1, extdStartSegP2)

    #         intersPt = perpendicularIntersectionPointToLine(extdEndSegP2,extdSegLineEq)

    #         newSegListPtStartOper = []

    #         #
    #         ptsList.append(intersPt)
    #         intersPt_extdStartIdx = len(ptsList) - 1 #last one added to the list

    #         for operIdx in range(0,len(operToApplyToSegList)):
    #             oper = operToApplyToSegList[operIdx]
    #             if oper is (Operations.KEEP or Operations.SEG_EXTEND_END):
    #                 newSegListPtStartOper.append(segList[operIdx])
    #                 continue

    #             if oper is Operations.SEG_EXTEND_START:
    #                 oldSeg: Segment = segList[operIdx]
    #                 newSeg = Segment(intersPt_extdStartIdx, oldSeg.endId)
    #                 newSegListPtStartOper.append(newSeg)
    #                 continue

    #             if oper is Operations.REPLACE:
    #                 newSeg = Segment(extendSegEndIdx, intersPt_extdStartIdx)
    #                 newSegListPtStartOper.append(newSeg)
    #                 continue

    #         return convertSegListToSimplGeomLS(newSegListPtStartOper, ptsList, edgeToBeSimplified)

    # # TODO: tune the algorithm such that both of START and END cases are treated at the same time
    # if ExtendNeighToPointStartFlag:
    #     # In this case, we have to remove the first segment in our LineString.
    #     # We will create a new segment from the starting point until the perpendicular to the segment to extend

    #     extendSegStartIdx = simplConfigOper[0][0]
    #     extendPointStartIdx = simplConfigOper[2][0]

    #     extendSegStart: Segment = segList[extendSegStartIdx]
    #     extendPointStart: Segment = segList[extendPointStartIdx]

    #     #retrieve coordinates
    #     # coords of the segment to be extended from the START
    #     extdStartSegP1: shpPoint = shpPtList[extendSegStart.startId]
    #     extdStartSegP2: shpPoint = shpPtList[extendSegStart.endId]

    #     extdStartPointP1: shpPoint = shpPtList[extendPointStart.startId]
    #     extdStartPointP2: shpPoint = shpPtList[extendPointStart.endId]

    #     extdSegLineEq: LineEquation = LineEquation(extdStartSegP1, extdStartSegP2)

    #     intersPt = perpendicularIntersectionPointToLine(extdStartPointP1,extdSegLineEq)

    #     #
    #     newSegListPtStartOper = []

    #     ptsList.append(intersPt)
    #     intersPt_extdStartIdx = len(ptsList) - 1 #last one added to the list

    #     for operIdx in range(0,len(operToApplyToSegList)):
    #         oper = operToApplyToSegList[operIdx]
    #         if oper is Operations.KEEP:
    #             newSegListPtStartOper.append(segList[operIdx])

    #         if oper is Operations.SEG_EXTEND_START:
    #             oldSeg: Segment = segList[operIdx]
    #             newSeg = Segment(intersPt_extdStartIdx, oldSeg.endId)
    #             newSegListPtStartOper.append(newSeg)

    #         if oper is Operations.PT_EXTEND_START:
    #             oldSeg: Segment = segList[operIdx]
    #             newSeg = Segment(oldSeg.startId, intersPt_extdStartIdx)
    #             newSegListPtStartOper.append(newSeg)

    #     return convertSegListToSimplGeomLS(newSegListPtStartOper, ptsList, edgeToBeSimplified)

    if ExtendNeighToPointEndFlag:
        pass