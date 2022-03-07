from cgitb import small
from tgap_ng.datastructure import PlanarPartition

from geopandas import GeoSeries
from shapely import geometry

from math import sqrt, pow

def createPoint(x,y) -> geometry.Point:
    return geometry.Point(x,y)

def computeLength(pt1: geometry.Point, pt2: geometry.Point):
    return sqrt(pow(pt2.x-pt1.x,2) + pow(pt2.y-pt1.y,2))

class Segment:
    """
    Class representing a connnection between two points, and that respective length
    """
    def __init__(self, start: geometry.Point, end: geometry.Point) -> None:
        self.start = start
        self.end = end
        self.length = computeLength(self.start,self.end)

class PointSequence:
    """
    Helper class, used for saving the points for a LineString/edge in sequence. 
    
    Written in the form of a doubly-linked list,  but where the
    connection between points has a weigth (i.e. distance)
    """
    def __init__(self, ptList) -> None:
        self.ptList = self.convertSimplPtsToShp(ptList)
        self.segList = []
        self.smallestSegmentId = 0
        self.smallestSegmentLen = 0
        for i in range(0,len(self.ptList)-2):
            seg = Segment(self.ptList[i], self.ptList[i+1])
            
            self.segList.append(seg)
            if seg.length > self.smallestSegmentLen:
                self.smallestSegmentId = len(self.segList) - 1 #since this segment is lastly saved in the list, the index is len-1
                self.smallestSegmentLen = seg.length
    
    def convertSimplPtsToShp(self, ptList) -> list:
        shpPtList = []
        for pt in ptList:
            shpPtList.append(createPoint(pt[0],pt[1]))

        return shpPtList

    def __str__(self) -> str:
        msg = ""
        for pt in self.ptList:
            msg += str(pt) + " "
        return msg

    def printSmallestSegment(self):
        print(f"Smallest segment has id {self.smallestSegmentId}, len: {self.smallestSegmentLen}, and goes from point"
              f"{self.segList[self.smallestSegmentId]}")

    # def getStartPosSmallestSegment(self):
    #     minPt = min(self.segList, key=lambda elem: elem.length)



    # NOTE: Just remove one segment for the time being, to be evaluated later
    # def returnXPercentileSmallestSegments(self, x: int):
    #     """
    #     Return the pair of points (pi,pj) which have their respective segment lengths in the
    #     lowest 20% of all the lengths in the LineString

    #     NOTE: Uses the pythagorean distance. Should I implement an alternative to it?
    #     """

    #     for i in range(0,len(self.ptList)-2):
    #         # Iterate from first element to second to last,
    #         # as our last connection will be from len-2 to len-1

    #         seg = Segment(self.ptList[i], self.ptList[i+1])


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
    ptsList = list(geom.coords)

    #shpPtsList : list(geometry.Point) = []
    #Create a point sequence containing the points and segments between them
    ptSeq = PointSequence(ptsList)
    #for pt in ptsList:

        #print(pt)
        #shpPtsList.append(geometry.Point(pt[0],pt[1]))

    ptSeq.printSmallestSegment()

    # for elem in ptSeq:
    #     print(elem)




    return [None, None]

def segmentLength():
    pass