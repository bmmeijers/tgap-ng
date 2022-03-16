from cgitb import small
from enum import IntEnum
from tgap_ng.datastructure import PlanarPartition

from geopandas import GeoSeries
from shapely import geometry

from simplegeom import geometry as simplgeom

import matplotlib.pyplot as plt

from math import sqrt, pow

class segmentOperations(IntEnum):
    KEEP = 0
    REMOVE = 1
    EXTEND_SEGMENT_START = 2
    EXTEND_SEGMENT_END = 3
    EXTEND_POINT_START = 4
    EXTEND_POINT_END = 4

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

#     return geometry.LineString(orderedPtList)

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

#     return geometry.Point(x4, y4)

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
    print("Entered SY Simplification Module - v2.0")

    print(f"Original Edge: {edgeToBeSimplified.geometry}")

    geom: geometry.LineString = gpdGeom[edgeToBeSimplified.id]

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
    operToApplyToSeg = [segmentOperations.KEEP] * len(segList)
    print(f"Initial Segment List: {operToApplyToSeg}")

    # Modify the list to determine how the modifications should be performed
    operToApplyToSeg[smlstSegId] = segmentOperations.REMOVE # the shortest segment will always be removed

    if len(segList) < 4:
        print("This polyline can't be simplified as it contains less than 4 segments")
        return
    if len(segList) == 4:
        # We have a special case here, where our ployline is short
        pass
    if len(segList)>4:
        try:
            # neighbour to the right should be removed
            operToApplyToSeg[smlstSegId+1] = segmentOperations.REMOVE
            try:
                operToApplyToSeg[smlstSegId+2] = segmentOperations.EXTEND_SEGMENT_START
                pass
            except:
                pass
        except IndexError as idxErr:
            # In this case, our shortest segment is the last one in the polyline. If the line is circular
            # then we can consider the first segment in our list to be removed (Idx 0), and the second one to be extended (Idx 1)
            pass


    #

def segmentLength():
    pass