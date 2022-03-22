from enum import IntEnum
from tgap_ng.datastructure import PlanarPartition, Edge

from geopandas import GeoSeries
from shapely.geometry import Point as shpPoint, LineString as shpLS

from simplegeom import geometry as simplgeom

import matplotlib.pyplot as plt

from tgap_ng.datastructure import angle

from math import sqrt, pow

class ModifySegmentOper(IntEnum):
    # Flag to signal that a segment needs to modify one of its endpoints
    EXTEND_START = 0
    EXTEND_END = 1

class ModifyPointOper(IntEnum):
    # Flag to signal that a segment will be replaced by the connection from
    # its Start/End point perpendicular on a segment
    EXTEND_START = 0
    EXTEND_END = 1

class GeneralSegmentOper(IntEnum):
    KEEP = 0
    REMOVE = 1
    REPLACE = 2

def createPoint(x,y) -> shpPoint:
    return shpPoint(x,y)

def computeLength(pt1, pt2):
    return sqrt(pow(pt2[0]-pt1[0],2) + pow(pt2[1]-pt1[1],2))

class Segment:
    """
    Class representing a connnection between two points, and that respective length

    considering a list PtList containing shapely Points, the segments contian the ids from that point
    TODO Simplify this
    """
    def __init__(self, startId: int, endId: int, initPtList = None)-> None:
        self.startId = startId
        self.endId = endId
        self.initPtList = initPtList
        self.length = -1
        if initPtList:
            #if we don't add initPtList, it means that we don't care about computing the length
            #self.length = computeLength(self.initPtList[self.startId],self.initPtList[self.endId])
            # shapely distance Point.distance(Point)
            self.length = self.initPtList[self.startId].distance(self.initPtList[self.endId])

class LineEquation:
    """
    Class used for storing the equation of a line in the form of y = m*x + b
    Where m is the slope and b is the y-intercept of the line
    """
    def __init__(self, *args) -> None:
        # We can initialize a Line Equation using either: 
        # - 2 Shapely points, and compute m and b or
        # - pass the values of m and b directly

        if len(args) < 2:
            print('NOT ENOUGH VARIABLES WERE PASSED TO THE OBJECT CONSTRUCTOR')

        arg1 = args[0]
        arg2 = args[1]

        # CONSTRUCTOR OVERLOADING
        if isinstance(arg1, shpPoint) and isinstance(arg2, shpPoint):
            # Slope = (y2 - y1)/(x2 - x1)
            # Y-intercept = (x1*(y1-y2))/(x2-x1) + y1
            self.slope = (arg2.y - arg1.y)/(arg2.x - arg1.x)
            self.yintercept = (arg1.x*(arg1.y - arg2.y))/(arg2.x - arg1.x) + arg1.y
        elif isinstance(arg1, float) and isinstance(arg2, float):
            self.slope = arg1
            self.yintercept = arg2
        else: 
            print("LineEquation oject was not constructed")

def intersectionPoint(line1: LineEquation, line2: LineEquation):
    # considering we have two line equations: y = m1*x + b1 and y = m2*x + b2
    # X_intersection = (b2 -b1)/(m1-m2) and Y_intersection = (m1*xintersection) + b2
    # extdStartSeg - Line 1 ; perpSeg - Line 2
    x = (line2.yintercept - line1.yintercept)/(line1.slope - line2.slope)
    y = line1.slope*x + line1.yintercept

    return (x,y)


def convertSimplPtsToShp(ptList) -> list:
        #get a list of coordinates like (x y) and convert them to Shapely Point
        shpPtList = []
        for pt in ptList:
            shpPtList.append(createPoint(pt[0],pt[1]))

        return shpPtList

def printSmallestSegment(self):
    print(f"Smallest segment has id {self.smlstSegId}, len: {self.smlstSegLen}")

# TODO: Change to include 
# def convertSegToShpLS(self):
#     # iterate the segments in order, and create a LineString containing the points in order
#     # use this method in the end, after 
#     orderedPtList = []
#     lineStartId = self.segList[0].startId #first point in the modified segList
#     orderedPtList.append(self.ptList[lineStartId]) #append the first point

#     for seg in self.segList:
#         orderedPtList.append(self.ptList[seg.endId])

#     return shpLS(orderedPtList)

def convertPtListToSimplGeomLS(ptList: list):
    # Convert our segments to LineString of type SimpleGeometry
    # TODO: copy-paste from above, implement DRY
    orderedPtList = []
    for pt in ptList:
        simplePt = simplgeom.Point(pt[0], pt[1], 28992)
        orderedPtList.append(simplePt)

    return simplgeom.LineString(orderedPtList)

# TODO: Adapt it to take a Segment and NOT A Polyline Structure
# def pointToLinePerpendicular(pointId, segment: Segment, lineStruct: PolylineStructure):
#     # the intersection point between a segment and the perpendicular from a point to that segment
#     # Based on: https://stackoverflow.com/questions/1811549/perpendicular-on-a-line-from-a-given-point
#     P1 = lineStruct.ptList[segment.startId] # Get P1 coords from list of points
#     P2 = lineStruct.ptList[segment.endId]

#     P3 = lineStruct.ptList[pointId]
#     dx = P2.x - P1.x
#     dy = P2.y - P1.y
#     mag = sqrt(dx*dx + dy*dy)
#     dx /= mag
#     dy /= mag

#     # translate the point and get the dot product
#     lambdaRes = (dx * (P3.x - P1.x)) + (dy * (P3.y - P1.y))
#     x4 = (dx * lambdaRes) + P1.x
#     y4 = (dy * lambdaRes) + P1.y

#     return shpPoint(x4, y4)

def plotShpLS(line: shpLS, color: str):
    x,y = line.xy
    plt.plot(x,y,c=color)
    plt.show()
    

def simplifySY(edgeToBeSimplified: Edge, pp: PlanarPartition, tolerance, DEBUG = False, gpdGeom = None):
    """
    Method used for simplifying a polyline having characteristics of man-made structures
    (i.e. orthogonal turns inbetween segments)

    Simplification based on the paper "Shape-Adaptive Geometric Simplification of Heterogeneous Line Datasets"
    by Samsonov and Yakimova. Further info on this algorithm can be found in 3.1.1. Simpl. of orthogonal segments.

    Topological awareness also needs to be accounted for, both in the case of self-intersections (which are mentioned
    in the paper) as well as with neighbouring structures (which is beyond the scope of that paper).
    """
    print("Entered SY Simplification Module - v2.0")

    print(f"Original Edge: {edgeToBeSimplified.geometry}")

    geom: shpLS = gpdGeom[edgeToBeSimplified.id]

    #plot the initial geometry
    plotShpLS(geom, "red")

    ptsList = list(geom.coords)

    # Create a point list containing the Shapely Points, and from that generate a segment List
    shpPtList = convertSimplPtsToShp(ptsList)
    segList = []
    isCircular = False

    if shpPtList[0] == shpPtList[-1]:
        #we have a closed edge (start Node = end Node)
        isCircular = True

    smlstSegId = -1 # first point will be replaces anyways,TODO we can add a check if it's -1 -> error
    smlstSegLen = 1000000 # max value

    for i in range(0,len(shpPtList)-1):
        # create segments from pairs of points, in order
        
        if i == len(shpPtList) - 1 and isCircular:
            # if our polyline is circular, we can consider the last segment to go from Idx_len-2 to Idx_0
            seg = Segment(i, 0, shpPtList)
        else:
            seg = Segment(i, i+1, shpPtList)
            
        segList.append(seg)

        if seg.length < smlstSegLen:
            # save the Idx and length of the smallest segment
            smlstSegId = len(segList) - 1 # since this segment is lastly saved in the list, the index is len-1
            smlstSegLen = seg.length

    print(f"Smallest segment has id: {smlstSegId} and length: {smlstSegLen}")

    # create a list which says if a segment should be removed, kept or extended
    operToApplyToSegList = [GeneralSegmentOper.KEEP] * len(segList)
    print(f"Initial Segment List: {operToApplyToSegList}")

    # Modify the list to determine how the modifications should be performed
    operToApplyToSegList[smlstSegId] = GeneralSegmentOper.REPLACE # the shortest segment will always be removed

    segmentListLength = len(segList)

    if isCircular and segmentListLength < 6:
        print("In the case of a circular edge, at least 6 segments are required to perform the simplification!")
        # Otherwise topo errors?
        return
    
    if segmentListLength < 4:
        print("This polyline can't be simplified as it contains less than 4 segments")
        return
    if segmentListLength == 4:
        # We have a special case here, where our ployline is short
        pass
    if segmentListLength > 4:
        # Construct the operation list

        #####
        # go to the RIGHT of our shortest segment
        try:
            # neighbour to the right should be removed
            operToApplyToSegList[smlstSegId+1] = GeneralSegmentOper.REMOVE

            try:
                # second degree neighbour to the right should be modified (its start point should change)
                operToApplyToSegList[smlstSegId+2] = ModifySegmentOper.EXTEND_START

            except IndexError as idxErr:
                # this means that our 1st degree neighbour (to the right) is the last in our polyline, 
                # if we have a circular polyline, extend the Idx_0 segment, otherwise connect from 
                # endpoint of the right point
                    if isCircular:
                        operToApplyToSegList[0] = ModifySegmentOper.EXTEND_START
                    else:
                        operToApplyToSegList[smlstSegId+1] = ModifyPointOper.EXTEND_END
        except IndexError as idxErr:
            # In this case, our shortest segment is the last one in the polyline. If the line is circular
            # then we can consider the first segment in our list to be removed (Idx 0), and the second one to be extended (Idx 1)
            if isCircular:
                operToApplyToSegList[0] = GeneralSegmentOper.REMOVE
                operToApplyToSegList[1] = ModifySegmentOper.EXTEND_START
            else:
                # Our segment is the last in the polyline
                operToApplyToSegList[smlstSegId] = ModifyPointOper.EXTEND_END
        
        #####
        # go to the LEFT of our shortest segment
        # Python considers negative indexes as going backwards in the list, so we have to force it in another way

        if smlstSegId-1 >= 0:
            # neighbour to the left should be removed
            operToApplyToSegList[smlstSegId-1] = GeneralSegmentOper.REMOVE

            if smlstSegId-2 >= 0:
                # second degree neighbour to the left should be modified (its start point should change)
                operToApplyToSegList[smlstSegId-2] = ModifySegmentOper.EXTEND_END

            else:
                # this means that our 1st degree neighbour (to the right) is the last in our polyline, 
                # if we have a circular polyline, extend the Idx_0 segment, otherwise connect from 
                # endpoint of the right point
                    if isCircular:
                        operToApplyToSegList[segmentListLength-1] = ModifySegmentOper.EXTEND_END
                    else:
                        operToApplyToSegList[smlstSegId-1] = ModifyPointOper.EXTEND_START            
        else:
            # This is the case where our shortest segment is the first one in the list (Idx_0)
            if isCircular:
                operToApplyToSegList[segmentListLength-1] = GeneralSegmentOper.REMOVE
                operToApplyToSegList[segmentListLength-2] = ModifySegmentOper.EXTEND_END
            else:
                # Our segment is the first in the polyline, so extend its starting point
                operToApplyToSegList[smlstSegId] = ModifyPointOper.EXTEND_START


    # Check that the configuration is correct
    # It should contain either two Segment Extensions (Start/End) + Replace or One Point Extension + Segment Extension
    segExtStartOperNo = 0
    segExtEndOperNo = 0
    ptExtStartOperNo = 0
    ptExtEndOperNo = 0
    replaceOperNo = 0

    for oper in operToApplyToSegList:
        # match/case now available in Python 3.10!
        if oper is ModifySegmentOper.EXTEND_START:
            segExtStartOperNo += 1
            continue
        elif oper is ModifySegmentOper.EXTEND_END:
            segExtEndOperNo += 1
            continue
        elif oper is ModifyPointOper.EXTEND_START:
            ptExtStartOperNo += 1
            continue
        elif oper is ModifyPointOper.EXTEND_END:
            ptExtEndOperNo += 1
            continue
        elif oper is GeneralSegmentOper.REPLACE:
            replaceOperNo += 1
            continue            

    simplConfig = [segExtStartOperNo, segExtEndOperNo, ptExtStartOperNo, ptExtEndOperNo, replaceOperNo]

    # Acceptable configurations: 1 Segment_Start, 1 Segment_End, 1 Replace (for Circular: only this is available!! - as we don't have any end points)
    # 1 Segment Start + 1 Point Start, no Replace OR 1 Segment End + 1 Point End, no Replace
    extendLeftRightNeighFlag = False
    ExtendNeighToPointFlag = False
    if simplConfig == [1,1,0,0,1]:
        print("We have to extend the 2nd degree neighbours (left/right) and replace the shortest segment with its perpendicular")
        extendLeftRightNeighFlag = True
    elif simplConfig == [1,0,1,0,0] or simplConfig == [0,1,0,1,0]:
        print("Determine the perpendicular from point to extension of segment")
        ExtendNeighToPointFlag = True
    else:
        print("SOMETHING WENT WRONG WITH CREATING A SIMPLIFICATION STRUCTURE")
        raise Exception

    if extendLeftRightNeighFlag:
        # in this case, we will replace our shortest segment with its perpendicular
        # determine its start by extending the segment_END to this new line
        # detemine its end by extending the segment_START to it

        # TODO: add checks to ensure that there are no self-intersections

        # The way checking worked before, the wrong indexes were returned
        for operIdx in range(0,len(operToApplyToSegList)):
            oper = operToApplyToSegList[operIdx]
            if oper is ModifySegmentOper.EXTEND_START:
                extendSegStartIdx = operIdx
                continue
            elif oper is ModifySegmentOper.EXTEND_END:
                extendSegEndIdx = operIdx
                continue
            elif oper is GeneralSegmentOper.REPLACE:
                replSegIdx = operIdx
                continue    

        extendSegStart: Segment = segList[extendSegStartIdx]
        extendSegEnd: Segment = segList[extendSegEndIdx]
        replSeg: Segment = segList[replSegIdx]

        #retrieve coordinates

        # coords of the segment to be extended from the START
        extdStartSegP1: shpPoint = shpPtList[extendSegStart.startId]
        extdStartSegP2: shpPoint = shpPtList[extendSegStart.endId]

        # coords of the segment to be extended from the END
        extdEndSegP1: shpPoint = shpPtList[extendSegEnd.startId]
        extdEndSegP2: shpPoint = shpPtList[extendSegEnd.endId]

        # coords of the segment to be REPLACED
        replSegP1: shpPoint = shpPtList[replSeg.startId]
        replSegP2: shpPoint = shpPtList[replSeg.endId]
        

        # Determine the line equation of extdStartSeg 
        extdStartSegLineEq = LineEquation(extdStartSegP1, extdStartSegP2)

        # Determine the line equation of extdEndSeg 
        extdEndSegLineEq = LineEquation(extdEndSegP1, extdEndSegP2)

        # REPALCE THE SHORTEST SEGMENT
        # Determine the slope of replSeg 
        replSeg_slope = (replSegP2.y - replSegP1.y)/(replSegP2.x - replSegP1.x)

        # determine the MIDPOINT of the segment to be replaced
        midPoint = shpPoint((replSegP1.x+replSegP2.x)/2,(replSegP1.y+replSegP2.y)/2)

        # the slope of the line PERPendicular to our replaced line is -(1/m)
        perpSeg_slope = -1/replSeg_slope
        perpSeg_yintercept = midPoint.y - perpSeg_slope*midPoint.x

        perpSegLineEq = LineEquation(perpSeg_slope, perpSeg_yintercept)

        # Intersect extdStartSeg & perpSeg to determine the intersection point
        newPoint_extdStartSeg = intersectionPoint(extdStartSegLineEq, perpSegLineEq)
        newPoint_extdEndSeg = intersectionPoint(extdEndSegLineEq, perpSegLineEq)

        newSegList = []

        ptsList.append(newPoint_extdStartSeg)
        newPoint_extdStartIdx = len(ptsList) - 1 #last one added to the list

        ptsList.append(newPoint_extdEndSeg)
        newPoint_extdEndIdx = len(ptsList) - 1 #last one added

        for operToApplyIdx in range(0,len(operToApplyToSegList)):
            operToApply = operToApplyToSegList[operToApplyIdx]
            # if it is GeneralSegmentOper.REMOVE do nothing
            if operToApply is GeneralSegmentOper.KEEP:
                newSegList.append(segList[operToApplyIdx])
            elif operToApply is GeneralSegmentOper.REPLACE:
                replacementSegment = Segment(newPoint_extdEndIdx, newPoint_extdStartIdx)
                newSegList.append(replacementSegment)
            elif operToApply is ModifySegmentOper.EXTEND_START:
                oldSegment: Segment = segList[operToApplyIdx]
                newSegment = Segment(newPoint_extdStartIdx, oldSegment.endId)
                newSegList.append(newSegment)
            elif operToApply is ModifySegmentOper.EXTEND_END:
                oldSegment: Segment = segList[operToApplyIdx]
                newSegment = Segment(oldSegment.startId, newPoint_extdEndIdx)
                newSegList.append(newSegment)

        # Generate a new LineString from newSegmentList
        newPtsList = []
        for segIdx in range(0,len(newSegList)):
            seg: Segment = newSegList[segIdx]
            if segIdx == len(newSegList)-1: 
                #if we have the last element, add both the end point and the first points
                newPtsList.append(ptsList[seg.startId])
                newPtsList.append(ptsList[seg.endId])
            else:
                newPtsList.append(ptsList[seg.startId])

        simplifiedLS: shpLS = shpLS(newPtsList)

        plotShpLS(simplifiedLS, "green")

        newGeom = convertPtListToSimplGeomLS(newPtsList)
        newEdge = Edge(
            edgeToBeSimplified.id,
            edgeToBeSimplified.start_node_id,
            angle(newGeom[0],newGeom[1]),
            edgeToBeSimplified.end_node_id,
            angle(newGeom[-1], newGeom[-2]),
            edgeToBeSimplified.left_face_id,
            edgeToBeSimplified.right_face_id,
            newGeom,
            {} #no info for now
        )
        return newEdge

