from .SY_constants import Operations, PositionRelevantPt
from shapely.geometry import Point as shpPoint

from __future__ import annotations #used for type-hinting for lists (like list[Class])

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

class AdvancedSegment:
    """
    A class which attaches information related to which operation to perform on it (in the context of SY Simplification),
    And which is the relevant point (start/end), on a specific Segment object
    """

    def __init__(self, seg: Segment, operation: Operations, location: PositionRelevantPt) -> None:
        self.seg = seg
        self.operation = operation
        self.location = location

    def setParameters(self, newOperation: Operations = None, newLocation: PositionRelevantPt = None):
        self.operation = newOperation
        self.location = newLocation

class SegmentCollention():
    """
    Basically a line formed by segments. 
    
    Takes in a simple list of points, and creates a list of segments and determines the opererations 
    which should be perfom on on each individual segments.

    It also determines which is the relevant/reference point on that respective segment, so that the appropiate transformation
    can be applied to it.

    It can also return its simplified version on request, or flag if there is a topological constraint
    """
    def __init__(self, pointList: list) -> None:
        self.segListSimple: list[Segment] = [] # doesn't contain any operation + also, we don't care to keep it
        self.isCircular = False
        self.shortestSegmentId = -1 # first point will be replaces anyways

        if pointList[0] == pointList[-1]:
            #we have a closed edge (start Node = end Node)
            self.isCircular = True

        # Generate list of segments
        for i in range(0,len(pointList)-1):
            # create segments from pairs of points, in order
            
            if i == len(pointList) - 1 and self.isCircular:
                # if our polyline is circular, we can consider the last segment to go from Idx_len-2 to Idx_0
                seg = Segment(i, 0, pointList)
            else:
                seg = Segment(i, i+1, pointList)
                
            self.segListSimple.append(seg)

            if seg.length < self.segListSimple[self.shortestSegmentId]:
                # if it is smaller, than save that new id
                self.shortestSegmentId = len(self.shortestSegmentId) - 1 # since this segment is lastly saved in the list, the index is len-1

        if False:
            print(f"Smallest segment has id: {self.shortestSegmentId} and length: {self.self.segListSimple[self.shortestSegmentId].length}")        
        
        try:
            self.segListAdv = self.attachAdvancedOperations(self.segListSimple, self.shortestSegmentId)
        except:
            print("Stopping the creation process of Segment Collection")
            return None
            # TODO: Create a check for this None in main

        # Since we're working exclusively with Shortcut simplification (except when our line has 4 segments)
        # The only operations which is required is determining the perpendicular from a point to a line equation.


    def attachAdvancedOperations(self, segList: list[Segment],  smlstSegId: int) -> list[AdvancedSegment]:
        """
        Generate a list containing advanced information related to the operations which can be applied to the respective segment.

        !!!WARNING!!!
        There's a lot of spaghetti-code below!
        """
        # Initialize the advanced list:
        segListAdvanced: list[AdvancedSegment] =  []
        for seg in segList:
            advSeg = AdvancedSegment(seg, Operations.Keep, PositionRelevantPt.NA)
            segListAdvanced.append(advSeg)
        
        lenSegList = len(segListAdvanced)

        # Here we perform some checks to see if it is possible to attach operations
        if self.isCircular and lenSegList < 6:
            print("In the case of a circular edge, at least 6 segments are required to perform the simplification!")
            # Otherwise topo errors?
            raise Exception

        if lenSegList < 4:
            print("This polyline can't be simplified as it contains less than 4 segments")
            raise Exception

        if lenSegList == 4:
            # We have a special case here, where our ployline is short
            #TODO: fix to work with short segments as well
            raise Exception

        # From this point, we can start applying the operations
        segListAdvanced[smlstSegId].setParameters(Operations.Remove)

        #####
        # go to the RIGHT of our shortest segment
        try:
            # neighbour to the right should be removed from the final geometry
            segListAdvanced[smlstSegId+1].setParameters(Operations.Ignore)

            try:
                # second degree neighbour to the right should be modified (its start point should change)
                segListAdvanced[smlstSegId+2].setParameters(Operations.Extend, PositionRelevantPt.Start)

            except IndexError as idxErr:
                # this means that our 1st degree neighbour (to the right) is the last in our polyline, 
                # if we have a circular polyline, extend the Idx_0 segment, otherwise connect from 
                # endpoint of the right point
                    if self.isCircular:
                        segListAdvanced[0].setParameters(Operations.Extend, PositionRelevantPt.Start)
                    else:
                        segListAdvanced[smlstSegId+1].setParameters(Operations.KeepRefPtOnly, PositionRelevantPt.End)
        except IndexError as idxErr:
            # In this case, our shortest segment is the last one in the polyline. If the line is circular
            # then we can consider the first segment in our list to be removed (Idx 0), and the second one to be extended (Idx 1)
            if self.isCircular:
                segListAdvanced[0].setParameters(Operations.Ignore)
                segListAdvanced[1].setParameters(Operations.Extend, PositionRelevantPt.Start)
            else:
                # Our segment is the last in the polyline
                segListAdvanced[smlstSegId].setParameters(Operations.KeepRefPtOnly, PositionRelevantPt.End)
        
        #####
        # go to the LEFT of our shortest segment
        # Python considers negative indexes as going backwards in the list, so we have to force it in another way
        if smlstSegId-1 >= 0:
            # neighbour to the left should be removed
            segListAdvanced[smlstSegId-1].setParameters(Operations.Ignore)

            if smlstSegId-2 >= 0:
                # second degree neighbour to the left should be modified (its start point should change)
                segListAdvanced[smlstSegId-2].setParameters(Operations.Extend, PositionRelevantPt.End)

            else:
                # this means that our 1st degree neighbour (to the right) is the last in our polyline, 
                # if we have a circular polyline, extend the Idx_0 segment, otherwise connect from 
                # endpoint of the right point
                    if self.isCircular:
                        segListAdvanced[-1].setParameters(Operations.Extend, PositionRelevantPt.End) # last segment in our list
                    else:
                        segListAdvanced[smlstSegId-1].setParameters(Operations.Extend, PositionRelevantPt.Start)
        else:
            # This is the case where our shortest segment is the first one in the list (Idx_0)
            if self.isCircular:
                segListAdvanced[-1].setParameters(Operations.Ignore)
                segListAdvanced[-2].setParameters(Operations.Extend, PositionRelevantPt.End)
            else:
                # Our segment is the first in the polyline, so extend its starting point
                segListAdvanced[smlstSegId].setParameters(Operations.KeepRefPtOnly, PositionRelevantPt.Start)

        return segListAdvanced

    def simplify():
        pass

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
            return

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