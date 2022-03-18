from enum import IntEnum
from tgap_ng.datastructure import PlanarPartition

from geopandas import GeoSeries
from shapely.geometry import Point as shpPoint, LineString as shpLS

from simplegeom import geometry as simplgeom

import matplotlib.pyplot as plt

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
    def __init__(self, startId: shpPoint, endId: int, initPtList = None)-> None:
        self.startId = startId
        self.endId = endId
        self.initPtList = initPtList
        self.length = -1
        if initPtList:
            #if we don't add initPtList, it means that we don't care about computing the length
            #self.length = computeLength(self.initPtList[self.startId],self.initPtList[self.endId])
            # shapely distance Point.distance(Point)
            self.length = self.initPtList[self.startId].distance(self.initPtList[self.endId])

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

def convertSegToSimplGeomLS(self):
    # Convert our segments to LineString of type SimpleGeometry
    # TODO: copy-paste from above, implement DRY
    orderedPtList = []
    lineStartId = self.segList[0].startId #first point in the modified segList

    #convert start point to simplegeometry Point
    firstPt = simplgeom.Point(self.ptList[lineStartId].x, self.ptList[lineStartId].y, 28992)
    orderedPtList.append(firstPt) #append the first point

    for seg in self.segList:
        endPointOfSeg = seg.endId
        simplePt = simplgeom.Point(self.ptList[endPointOfSeg].x, self.ptList[endPointOfSeg].y, 28992)
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
    

def simplifySY(edgeToBeSimplified, pp: PlanarPartition, tolerance, DEBUG = False, gpdGeom = None):
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
    operToApplyToSeg = [GeneralSegmentOper.KEEP] * len(segList)
    print(f"Initial Segment List: {operToApplyToSeg}")

    # Modify the list to determine how the modifications should be performed
    operToApplyToSeg[smlstSegId] = GeneralSegmentOper.REPLACE # the shortest segment will always be removed

    segmentListLength = len(segList)

    if isCircular and segmentLength < 6:
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
            operToApplyToSeg[smlstSegId+1] = GeneralSegmentOper.REMOVE

            try:
                # second degree neighbour to the right should be modified (its start point should change)
                operToApplyToSeg[smlstSegId+2] = ModifySegmentOper.EXTEND_START

            except IndexError as idxErr:
                # this means that our 1st degree neighbour (to the right) is the last in our polyline, 
                # if we have a circular polyline, extend the Idx_0 segment, otherwise connect from 
                # endpoint of the right point
                    if isCircular:
                        operToApplyToSeg[0] = ModifySegmentOper.EXTEND_START
                    else:
                        operToApplyToSeg[smlstSegId+1] = ModifyPointOper.EXTEND_END
        except IndexError as idxErr:
            # In this case, our shortest segment is the last one in the polyline. If the line is circular
            # then we can consider the first segment in our list to be removed (Idx 0), and the second one to be extended (Idx 1)
            if isCircular:
                operToApplyToSeg[0] = GeneralSegmentOper.REMOVE
                operToApplyToSeg[1] = ModifySegmentOper.EXTEND_START
            else:
                # Our segment is the last in the polyline
                operToApplyToSeg[smlstSegId] = ModifyPointOper.EXTEND_END
        
        #####
        # go to the LEFT of our shortest segment
        # Python considers negative indexes as going backwards in the list, so we have to force it in another way

        if smlstSegId-1 >= 0:
            # neighbour to the left should be removed
            operToApplyToSeg[smlstSegId-1] = GeneralSegmentOper.REMOVE

            if smlstSegId-2 >= 0:
                # second degree neighbour to the left should be modified (its start point should change)
                operToApplyToSeg[smlstSegId-2] = ModifySegmentOper.EXTEND_END

            else:
                # this means that our 1st degree neighbour (to the right) is the last in our polyline, 
                # if we have a circular polyline, extend the Idx_0 segment, otherwise connect from 
                # endpoint of the right point
                    if isCircular:
                        operToApplyToSeg[segmentListLength-1] = ModifySegmentOper.EXTEND_END
                    else:
                        operToApplyToSeg[smlstSegId-1] = ModifyPointOper.EXTEND_START            
        else:
            # This is the case where our shortest segment is the first one in the list (Idx_0)
            if isCircular:
                operToApplyToSeg[segmentListLength-1] = GeneralSegmentOper.REMOVE
                operToApplyToSeg[segmentListLength-2] = ModifySegmentOper.EXTEND_END
            else:
                # Our segment is the first in the polyline, so extend its starting point
                operToApplyToSeg[smlstSegId] = ModifyPointOper.EXTEND_START


    # Check that the configuration is correct
    # It should contain either two Segment Extensions (Start/End) + Replace or One Point Extension + Segment Extension

    segExtStartOperNo = len([i for i, e in enumerate(operToApplyToSeg) if e == ModifySegmentOper.EXTEND_START]) # count how many Extend_segment_starts we have
    segExtEndOperNo = len([i for i, e in enumerate(operToApplyToSeg) if e == ModifySegmentOper.EXTEND_END])

    ptExtStartOperNo = len([i for i, e in enumerate(operToApplyToSeg) if e == ModifyPointOper.EXTEND_START])
    ptExtEndOperNo = len([i for i, e in enumerate(operToApplyToSeg) if e == ModifyPointOper.EXTEND_END])
    replaceOperNo = len([i for i, e in enumerate(operToApplyToSeg) if e == GeneralSegmentOper.REPLACE])

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
        extendSegStartIdx = operToApplyToSeg.index(ModifySegmentOper.EXTEND_START)
        extendSegEndIdx = operToApplyToSeg.index(ModifySegmentOper.EXTEND_END)

        replSegIdx = operToApplyToSeg.index(GeneralSegmentOper.REPLACE)

        

        

def segmentLength():
    pass