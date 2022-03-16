from cgitb import small
from enum import IntEnum
from tgap_ng.datastructure import PlanarPartition

from geopandas import GeoSeries
from shapely import geometry

from simplegeom import geometry as simplgeom

import matplotlib.pyplot as plt

from math import sqrt, pow

class endSegCath(IntEnum):
    START_SEG = 0
    FINAL_SEG = 1
    SECOND_SEG = 2
    SECOND_TO_LAST = 3

def createPoint(x,y) -> geometry.Point:
    return geometry.Point(x,y)

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
            self.length = computeLength(self.initPtList[self.startId],self.initPtList[self.endId])

    # NOT NEEDED
    # def changeStartId(self, newStartId):
    #     self.startId = newStartId
    #     # TODO: SHOULD LENGTH BE RECOMPUTED? WE DON'T USE IT AFTERWARDS?

    # def changeEndId(self, newEndId):
    #     self.endId = newEndId

class PolylineStructure:
    """
    Helper class, used for saving the points for a LineString/edge in sequence. 
    
    Has a point List (containing all points which form the Polyline in order),
    segment list contianing segments (pi, pj)
    """
    def __init__(self, ptList) -> None:
        self.ptList = self.convertSimplPtsToShp(ptList)
        self.segList = []

        self.smallestSegmentId = -1 # first point will be replaces anyways,TODO we can add a check if it's -1 -> error
        self.smallestSegmentLen = 1000000 #max value

        for i in range(0,len(self.ptList)-1):
            #create segments from pairs of points, in order
            seg = Segment(i, i+1, ptList)
            self.segList.append(seg)

            if seg.length < self.smallestSegmentLen:
                self.smallestSegmentId = len(self.segList) - 1 #since this segment is lastly saved in the list, the index is len-1
                self.smallestSegmentLen = seg.length
    
    def convertSimplPtsToShp(self, ptList) -> list:
        #get a list of coordinates like (x y) and convert them to Shapely Point
        shpPtList = []
        for pt in ptList:
            shpPtList.append(createPoint(pt[0],pt[1]))

        return shpPtList

    def __str__(self) -> str:
        msg = ""
        for pt in self.ptList:
            msg += str(pt) + " "
        return msg

    def isSmlstSegEnd(self) -> endSegCath:
        # returns either the category (start, finish) if the smallest segment is an end one
        # or None is the segment is inside
        if self.smallestSegmentId == 0: #first regment in the polyline
            return endSegCath.START_SEG
        elif self.smallestSegmentId == len(self.segList) - 1: #last segment
            return endSegCath.FINAL_SEG
        return None


    def printSmallestSegment(self):
        print(f"Smallest segment has id {self.smallestSegmentId}, len: {self.smallestSegmentLen}")

    def convertSegToShpLS(self):
        #iterate the segments in order, and create a LineString containing the points in order
        orderedPtList = []
        lineStartId = self.segList[0].startId #first point in the modified segList
        orderedPtList.append(self.ptList[lineStartId]) #append the first point

        for seg in self.segList:
            orderedPtList.append(self.ptList[seg.endId])

        return geometry.LineString(orderedPtList)

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


def pointToLinePerpendicular(pointId, segment: Segment, lineStruct: PolylineStructure):
    # the intersection point between a segment and the perpendicular from a point to that segment
    # Based on: https://stackoverflow.com/questions/1811549/perpendicular-on-a-line-from-a-given-point
    P1 = lineStruct.ptList[segment.startId] # Get P1 coords from list of points
    P2 = lineStruct.ptList[segment.endId]

    P3 = lineStruct.ptList[pointId]
    dx = P2.x - P1.x
    dy = P2.y - P1.y
    mag = sqrt(dx*dx + dy*dy)
    dx /= mag
    dy /= mag

    # translate the point and get the dot product
    lambdaRes = (dx * (P3.x - P1.x)) + (dy * (P3.y - P1.y))
    x4 = (dx * lambdaRes) + P1.x
    y4 = (dy * lambdaRes) + P1.y

    return geometry.Point(x4, y4)

def plotShpLS(line: geometry.LineString, color: str):
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
    print("Entered SY Simplification Module")

    # for line in edgeToBeSimplified.geometry:
    #     print("Lines:" ,line)

    #convert geometry.wkt to Shapely LineString object
    #geom =  edgeToBeSimplified.geometry
    print(f"Original Edge: {edgeToBeSimplified.geometry}")

    geom: geometry.LineString = gpdGeom[edgeToBeSimplified.id]

    #plot the initial geometry
    plotShpLS(geom, "red")

    ptsList = list(geom.coords)

    # Create a point sequence containing the points and segments between them
    lineStruct = PolylineStructure(ptsList)

    lineStruct.printSmallestSegment()

    # Remove smallest segment
    smlSegId = lineStruct.smallestSegmentId
    numSeg = len(lineStruct.segList)

    isEndSeg = lineStruct.isSmlstSegEnd()
    if numSeg < 4:
        print(f"Polyline {geom} can't be simplifed any further, as it already has less than 4 segments")
        return # nothing we can do more, continue to next line
    if isEndSeg is not None:
        # ENDPOINT Simplification from SY
        # We have to simplify either the first or the last segment

        if isEndSeg == endSegCath.START_SEG:
            # remove neighbour from the right, modify start coordinate of 2nd degree neighbour
            # by tracing the perpendicular from the start of the removed segment to it

            #Remove the first and second elements from the segment list
            startRemovedSegIndex = lineStruct.segList[0].startId
            secondDegNeighRight: Segment = lineStruct.segList[2] # TODO: HARDCODED it should be 2 to the right, modify this!

            del lineStruct.segList[0]
            del lineStruct.segList[0]

            intersectionPoint = pointToLinePerpendicular(startRemovedSegIndex, secondDegNeighRight, lineStruct)
            
            # add the intersectionPoint to our ptList, and create a new segment
            # with start = startRemovedSegIndex, end = intersection point
            lineStruct.ptList.append(intersectionPoint)
            intersPtIdx = len(lineStruct.ptList) -1 #index of new point - last position in list

            newSegStart = Segment(startRemovedSegIndex, intersPtIdx)

            # MODIFY THE START POINT OF 2nd DEG NEIGHBOUR 
            # (TODO: which is now the first segment in our list, this seems really hardcoded, should fix this)
            lineStruct.segList[0].startId = intersPtIdx

            # Append newSeg to start of segList
            lineStruct.segList.append(0, newSegStart)

            return

        if isEndSeg == endSegCath.FINAL_SEG:
            # remove last segment, segment to the left and modify end-point of 2nd degree neighbour

            #Remove the first and second elements from the segment list
            endRemovedSegIndex = lineStruct.segList[-1].endId
            secondDegNeighLeft: Segment = lineStruct.segList[-3] # TODO: HARDCODED it should be 2 to the left, modify this!

            #remove last two elements
            lineStruct.segList.pop()
            lineStruct.segList.pop()

            intersectionPoint = pointToLinePerpendicular(endRemovedSegIndex, secondDegNeighLeft, lineStruct)

            # add the intersectionPoint to our ptList, and create a new segment
            # with start = startRemovedSegIndex, end = intersection point
            lineStruct.ptList.append(intersectionPoint)
            intersPtIdx = len(lineStruct.ptList) -1 #index of new point - last position in list

            newSegEnd = Segment(intersPtIdx, endRemovedSegIndex)

            # MODIFY THE START POINT OF 2nd DEG NEIGHBOUR 
            # (TODO: which is now the LAST segment in our list, this seems really hardcoded, should fix this)
            lineStruct.segList[-1].endId = intersPtIdx

            # Append newSeg to end of segList
            lineStruct.segList.append(newSegEnd)

            newGeom = lineStruct.convertSegToShpLS()
            plotShpLS(newGeom, "green")

            # change GPDGeom to new geometry
            gpdGeom[edgeToBeSimplified.id] = newGeom

            #Convert back to linestring and return it
            return lineStruct.convertSegToSimplGeomLS(), tolerance
    if numSeg == 4:
        #SHORT SIMPLIFICATION from SY Algorithm

        # This is the case where the segment which has to be removed is NOT an end segment, but our polylines contains only
        # 4 segments (i.e. seg to be removed either 1 or 2 out of [0,1,2,3])

        #simply remove 1 & 2 and create new line start in 0.end and end in 3.start
        return
    
    #In any other case, we simplify either Using Shortcut or Median
    
    #Remove segment to be simplified, alongside left/right neighbour
    secondDegNeighLeft = lineStruct[smlSegId-2]
    secondDegNeighRight = lineStruct[smlSegId+2]

    #Shortcut - considering we keep left Neigh intact and modify the start of the right Neigh
    intersectionPoint = pointToLinePerpendicular(secondDegNeighLeft.endId, secondDegNeighRight, lineStruct)


    return [None, None]

def segmentLength():
    pass