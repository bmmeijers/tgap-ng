from __future__ import annotations #used for type-hinting for lists (like list[Class])

from .SY_constants import (
    Operations, PositionRelevantPt, Direction, 
    PreClassificationException, ObjectNotCreatedException, Transaction,
    TopologyIssuesException, VerticalIntersectionNotImplemented
)
from .SY_utils import (
    safelyAppendToDict, decideBias, plotShpLS, safelyAddShpPtToQuadTree, 
    safelyRemoveShpPtFromQuadTree, quadTreeRangeSearchAdapter
)
from shapely.geometry import Point as shpPoint, LineString as shpLS

from simplegeom import geometry as simplgeom
from tgap_ng.datastructure import Edge, angle

from quadtree import QuadTree

class QuadTreeTransactionalManager:
    def __init__(self) -> None:
        self.trx = []

    def addTransaction(self, trans: Transaction, pt: shpPoint):  
        if [trans, pt] not in self.trx:      
            self.trx.append([trans, pt])

    def rollback(self, qt: QuadTree):
        for t in self.trx:
            if t[0] is Transaction.Remove:
                safelyAddShpPtToQuadTree(qt, t[1])
            elif t[0] is Transaction.Add:
                safelyRemoveShpPtFromQuadTree(qt, t[1])

    def returnShpPts(self) -> list[shpPoint]:
        ptsList = []
        for t in self.trx:
            ptsList.append(t[1])

        return ptsList

    def returnSimplPts(self) -> list:
        ptsList = []
        for t in self.trx:
            ptsList.append((t[1].x, t[1].y))

        return ptsList

class Segment:
    """
    Class representing a connnection between two points, and that respective length

    considering a list PtList containing shapely Points, the segments contian the ids from that point
    TODO Simplify this
    """
    def __init__(self, startId: int, endId: int, initPtList: list[shpPoint] = None)-> None:
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
    def __init__(self, pointList: list[shpPoint], planarPartition, originalEdge: Edge) -> None:
        self.segListSimple: list[Segment] = [] # doesn't contain any operation
        self.isCircular = False
        self.shortestSegmentId = -1 # first point will be replaces anyways
        self.pp = planarPartition
        self.origEdge = originalEdge

        if pointList[0] == pointList[-1]:
            #we have a closed edge (start Node = end Node)
            self.isCircular = True

        shortestLength = 10000000
        # Generate list of segments
        for i in range(0,len(pointList)-1):
            # create segments from pairs of points, in order
            
            if i == len(pointList) - 1 and self.isCircular:
                # if our polyline is circular, we can consider the last segment to go from Idx_len-2 to Idx_0
                seg = Segment(i, 0, pointList)
            else:
                seg = Segment(i, i+1, pointList)
                
            self.segListSimple.append(seg)

            if seg.length < shortestLength:
                # if it is smaller, than save that new id
                self.shortestSegmentId = len(self.segListSimple) - 1 # since this segment is lastly saved in the list, the index is len-1
                shortestLength = seg.length

        #print(f"Smallest segment has id: {self.shortestSegmentId} and length: {self.segListSimple[self.shortestSegmentId].length}")        
        
        try:
            self.segListAdv = self.attachAdvancedOperations(self.segListSimple, self.shortestSegmentId)
        except:
            #print(f"Stopping the creation process of Segment Collection for edge {pointList}")
            raise ObjectNotCreatedException

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
            #print("This polyline can't be simplified as it contains less than 4 segments")
            raise Exception

        if lenSegList == 4 and (smlstSegId == 1 or smlstSegId == 2):
            # If it is an interior segment in a short polyline, then we apply a special kind of flag,
            # which simply removes the interior point from the QuadTree and connects the first and last segments

            # if it is short but an ending-segment, then it goes in the normal shortcut algorithm
            segListAdvanced[smlstSegId].setParameters(Operations.ShortInterior)
            return segListAdvanced

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
                        segListAdvanced[smlstSegId-1].setParameters(Operations.KeepRefPtOnly, PositionRelevantPt.Start)
        else:
            # This is the case where our shortest segment is the first one in the list (Idx_0)
            if self.isCircular:
                segListAdvanced[-1].setParameters(Operations.Ignore)
                segListAdvanced[-2].setParameters(Operations.Extend, PositionRelevantPt.End)
            else:
                # Our segment is the first in the polyline, so extend its starting point
                segListAdvanced[smlstSegId].setParameters(Operations.KeepRefPtOnly, PositionRelevantPt.Start)

        return segListAdvanced

    def simplify(self, initialShpPtList: list[shpPoint], initialSimplePtList: list):
        simplTRX = QuadTreeTransactionalManager()

        # Besides recording the points which are deleted and added in the QuadTree in the transactional manager, 
        # we should also record them to check if there are any point contained withing the rectangle created by there recorded points
        ptsToVerifyTopology = [] 

        # Since we're only applying Shortcut and Diagonal simplifications, we require a Point and a Line, so that we can 
        # determine the perpendicular from the point on the line equation. This is common for all operations, and this can be easily
        # extracted as we indicate , with the operation Flag, which objects to take 
        refPointId = None
        refLineId  = None

        # dictionary which stores an operation as key and a list containing the ids of the objects which do that operation 
        segOperDict = {}
        for advSegId in range(0,len(self.segListAdv)):
            advSeg = self.segListAdv[advSegId]
            
            safelyAppendToDict(segOperDict, anyKey= advSeg.operation,anyObject= advSegId)

        # Pre-simplification checks
        # We should have either two extends or one extend and one KeepRefPtOnly
        isDiagonalSimpl = False

        if Operations.ShortInterior in segOperDict:
            isDiagonalSimpl = True
        elif len(segOperDict.get(Operations.Extend,[])) == 2 and Operations.KeepRefPtOnly not in segOperDict:
            # This is the case where we would have applied Median simplification, but we will replace that with the simple
            # point to line operation. 
            bias = decideBias()
            extendIdsList = segOperDict[Operations.Extend]

            # Search the ids with extend start and end
            idExtendStart = next((idx for idx in extendIdsList if self.segListAdv[idx].location == PositionRelevantPt.Start), None)
            idExtendEnd = next((idx for idx in extendIdsList if self.segListAdv[idx].location == PositionRelevantPt.End), None)
            if bias is Direction.Left:
                self.segListAdv[idExtendEnd].operation = Operations.KeepWithAnchorPt
                refPointId = idExtendEnd
                refLineId = idExtendStart
            else:
                self.segListAdv[idExtendStart].operation = Operations.KeepWithAnchorPt
                refPointId = idExtendStart
                refLineId = idExtendEnd
        elif len(segOperDict.get(Operations.Extend,[])) == 1 and len(segOperDict.get(Operations.KeepRefPtOnly,[])) == 1:
            # Normal shortcut operation
            refPointId = segOperDict[Operations.KeepRefPtOnly][0] # first (and only) element
            refLineId = segOperDict[Operations.Extend][0]
        else:
            raise PreClassificationException

        if not isDiagonalSimpl:
            # if it is a short simplification, no intersection is computed
            intersPt = intersectionFromAdvSegment(self.segListAdv[refPointId], self.segListAdv[refLineId] ,initialShpPtList)
            if intersPt is None:
                raise VerticalIntersectionNotImplemented
            intersPtShp: shpPoint = shpPoint(intersPt[0],intersPt[1])

            # append new Point to both lists
            initialShpPtList.append(intersPtShp)
            initialSimplePtList.append(intersPt)

            intersPtId = len(initialShpPtList) -1

            # Add intersection Point to QuadTree and transaction
            simplTRX.addTransaction(Transaction.Add, intersPtShp)

            safelyAddShpPtToQuadTree(self.pp.quadtree,intersPtShp)

        ##################################
        # From this point, we begin to do the actual simplification. All (important) operations which can be performed on a segment 
        # interact in some way or another with the intersection point. The following operations will be performed per cathegory:
        # Keep: 
        #   -> Add it to the list of Simplified Segments
        # Ignore:
        #   -> This will be skipped, no nothing on it
        # Remove:
        #   -> Not added to the final list
        #   -> Remove BOTH endpoint from the QuadTree (& add to trx)
        # Extend:
        #   -> replace the anchor Point with the intersection Point
        #   -> remove the initial anchor Pt from the QT, add new anchorPt to QT
        # KeepRefPtOnly:
        #   -> The anchor point will remain in place, and a new segment will be
        #       appended from the anchor point to intersPt
        #   -> Remove the point opposite of the anchorPt from QT
        # KeepWithAnchorPt: (occurs only when substituting Median simpl to Shortcut)
        #   -> add it to the new segment list, and before or after it (depending on the position of the relevant Point)
        #       add a new segment containing the IntersectionPt and AnchorPt

        simplifiedSegList = []

        if isDiagonalSimpl:
            # In this situation, we have a short polyline (4 segments), and we have to simplify an interior segment (id 1 or 2)
            # Connect Seg[0] with Seg[3] by using an intermediate direct segment [Seg[0].endId, Seg[3].startId]
            firstSeg = self.segListSimple[0]
            lastSeg = self.segListSimple[3]

            newSeg = Segment(firstSeg.endId, lastSeg.startId)

            newSegList = [firstSeg, newSeg, lastSeg]

            # From Quad Tree, we can remove either the endPt of seg[1] or StartPt of Seg[2] - same point!
            removedPtId = self.segListSimple[1].endId
            removedPt = initialShpPtList[removedPtId]

            simplTRX.addTransaction(Transaction.Remove, removedPt)
            safelyRemoveShpPtFromQuadTree(self.pp.quadtree, removedPt)

            simplTRX.addTransaction(Transaction.UseOnlyForCheck, initialShpPtList[firstSeg.endId])
            simplTRX.addTransaction(Transaction.UseOnlyForCheck, initialShpPtList[firstSeg.startId])

            # COPIED FROM BELOW, NOT SURE HOW TO SKIP OVER ALL THE OTHER OPERATIONS:
            pointsInModifiedArea = quadTreeRangeSearchAdapter(simplTRX.returnShpPts(),self.pp.quadtree)

            for discoveredPt in pointsInModifiedArea:
                #it might find points which we have added to QT, ignore those
                if discoveredPt in simplTRX.returnSimplPts():
                    pointsInModifiedArea.remove(discoveredPt)
            if len(pointsInModifiedArea) > 0:
                # TOPOLOGY ERRORS!!!!
                simplTRX.rollback(self.pp.quadtree)
                raise TopologyIssuesException

            return convertSegListToSimplGeomLS(newSegList, initialSimplePtList, self.origEdge)

        for iterAdvSegId in range(0,len(self.segListAdv)):
            currentAdvSeg: AdvancedSegment = self.segListAdv[iterAdvSegId]
            advSegOper = currentAdvSeg.operation
            if advSegOper is Operations.Keep:
                # Simply append to list
                simplifiedSegList.append(self.segListSimple[iterAdvSegId])
            elif advSegOper is Operations.Ignore:
                # DO NOTHING HERE
                pass
            elif advSegOper is Operations.Remove:
                startPt = initialShpPtList[advSeg.seg.startId]
                endPt = initialShpPtList[advSeg.seg.endId]

                # save to rollback later
                simplTRX.addTransaction(Transaction.Remove, startPt)
                simplTRX.addTransaction(Transaction.Remove, endPt)

                safelyRemoveShpPtFromQuadTree(self.pp.quadtree, startPt)
                safelyRemoveShpPtFromQuadTree(self.pp.quadtree, endPt)
            elif advSegOper is Operations.Extend:
                oldSeg = currentAdvSeg.seg
                if currentAdvSeg.location is PositionRelevantPt.Start:
                    # Start will be modified to take the new intersection Point
                    newSeg = Segment(intersPtId, oldSeg.endId)
                    startPt = initialShpPtList[oldSeg.startId]
                    simplifiedSegList.append(newSeg)

                    simplTRX.addTransaction(Transaction.Remove, startPt)
                    safelyRemoveShpPtFromQuadTree(self.pp.quadtree, startPt)
                else:
                    newSeg = Segment(oldSeg.startId, intersPtId)
                    endPt = initialShpPtList[oldSeg.endId]
                    simplifiedSegList.append(newSeg)

                    simplTRX.addTransaction(Transaction.Remove, endPt)
                    safelyRemoveShpPtFromQuadTree(self.pp.quadtree, endPt)


            elif advSegOper is Operations.KeepRefPtOnly:
                oldSeg = currentAdvSeg.seg
                if currentAdvSeg.location is PositionRelevantPt.Start:
                    newSeg = Segment(oldSeg.startId, intersPtId)
                    endPt = initialShpPtList[oldSeg.endId]
                    simplifiedSegList.append(newSeg)

                    simplTRX.addTransaction(Transaction.Remove, endPt)
                    safelyRemoveShpPtFromQuadTree(self.pp.quadtree, endPt)
                else:
                    # In this case, our anchor point is End
                    newSeg = Segment(intersPtId, oldSeg.endId)
                    startPt = initialShpPtList[oldSeg.startId]
                    simplifiedSegList.append(newSeg)

                    simplTRX.addTransaction(Transaction.Remove, startPt)
                    safelyRemoveShpPtFromQuadTree(self.pp.quadtree, startPt)
            elif advSegOper is Operations.KeepWithAnchorPt:
                oldSeg = currentAdvSeg.seg
                if currentAdvSeg.location is PositionRelevantPt.Start:
                    newSeg = Segment(intersPtId, oldSeg.startId)
                    simplifiedSegList.append(newSeg)
                    simplifiedSegList.append(oldSeg)
                else:
                    newSeg = Segment(oldSeg.endId, intersPtId)
                    simplifiedSegList.append(oldSeg)
                    simplifiedSegList.append(newSeg)

        # TOPOLOGICAL CHECK
        # Now that we have a list of simplified segments, we want to check if the shape formed by the points which have 
        # been added to the transaction manager doesn't contain any points in quadtree
        pointsInModifiedArea = quadTreeRangeSearchAdapter(simplTRX.returnShpPts(),self.pp.quadtree)

        # for discoveredPt in pointsInModifiedArea:
        #     #it might find points which we have added to QT, ignore those
        #     if discoveredPt in simplTRX.returnSimplPts():
        #         pointsInModifiedArea.remove(discoveredPt)
        # if len(pointsInModifiedArea) > 0:
        #     # TOPOLOGY ERRORS!!!!
        #     simplTRX.rollback(self.pp.quadtree)
        #     raise TopologyIssuesException

        # print("Simplification performed successfully")
        # return convertSegListToSimplGeomLS(simplifiedSegList, initialSimplePtList, self.origEdge)

##################################################################
# OTHER GEOMETRY STUFF
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

        self.isHorizontal = False
        self.isVertical = False

        # CONSTRUCTOR OVERLOADING
        if isinstance(arg1, shpPoint) and isinstance(arg2, shpPoint):
            # Slope = (y2 - y1)/(x2 - x1)
            # Y-intercept = (x1*(y1-y2))/(x2-x1) + y1

            delta_x = arg2.x - arg1.x
            delta_y = arg2.y - arg1.y
            if delta_x == 0:
                # Line will be Vertical, slope - undefined
                self.slope = None
                self.yintercept = None
                self.isVertical = True
                
                # CONVERT TO NORMAL FORM
                # A = y1-y2; B= x2-x1; C=x1*y2-x2*y1
                self.A = arg1.y - arg2.y 
                self.B = arg1.x - arg2.x # WILL BE 0
                self.C = arg1.x*arg2.y - arg2.x*arg1.y
            elif delta_y == 0:
                # Line will be Horizontal, slope - 0
                self.slope = 0
                self.yintercept = arg1.y #since the first part of the operation is 0 anyways
                self.isHorizontal = True
            else:
                self.slope = (delta_y)/(delta_x)
                self.yintercept = (arg1.x*(arg1.y - arg2.y))/(arg2.x - arg1.x) + arg1.y
        elif isinstance(arg1, float) and isinstance(arg2, float):
            if arg1 == 0:
                self.isHorizontal = True
            elif arg1 is None:
                self.isVertical = True
            self.slope = arg1
            self.yintercept = arg2
        else: 
            print("LineEquation oject was not constructed")

# MISC METHODS
# def intersectionPoint(line1: LineEquation, line2: LineEquation):
#     # considering we have two line equations: y = m1*x + b1 and y = m2*x + b2
#     # X_intersection = (b2 -b1)/(m1-m2) and Y_intersection = (m1*xintersection) + b2
#     # extdStartSeg - Line 1 ; perpSeg - Line 2    
#     x = (line2.yintercept - line1.yintercept)/(line1.slope - line2.slope)
#     y = line1.slope*x + line1.yintercept

#     return (x,y)


def convertPtListToSimplGeomLS(ptList: list):
    # Convert our segments to LineString of type SimpleGeometry
    orderedPtList = []
    for pt in ptList:
        simplePt = simplgeom.Point(pt[0], pt[1], 28992)
        orderedPtList.append(simplePt)

    return simplgeom.LineString(orderedPtList)

def convertToNormalForm(line: LineEquation):
    # takes a line in the form y = slope*x + y_intercept and tarnsform it into normal form by:
    # y - slope*x - y_intercept = 0, where a=1, b=-slope, c=-y_intercept
    a = 1
    b = -1*line.slope
    c = -1*line.yintercept

    return a,b,c 

def intersectionFromAdvSegment(ptSeg: AdvancedSegment, lineSeg: AdvancedSegment, ptsList: list[shpPoint]):
    lineEq = LineEquation(ptsList[lineSeg.seg.startId], ptsList[lineSeg.seg.endId])
    if ptSeg.location == PositionRelevantPt.End:
        pt = ptsList[ptSeg.seg.endId]
    else:
        pt = ptsList[ptSeg.seg.startId]

    return perpendicularIntersectionPointToLine(pt,lineEq)

def intersectionPoint(line1: LineEquation, line2: LineEquation):
    # considering we have two line equations: y = m1*x + b1 and y = m2*x + b2
    # X_intersection = (b2 -b1)/(m1-m2) and Y_intersection = (m1*xintersection) + b2
    # extdStartSeg - Line 1 ; perpSeg - Line 2

    if line1.slope == line2.slope:
        print("Lines are parallel, there is no intersection point")
        return None
    elif line2.isVertical:
        # We compute the intersection point using the noral form now
        # Convert also line2 to normal form:
        line1_a, line1_b, line1_c = convertToNormalForm(line1)

        # x = (-l2_c)/l2_a; y = (l1_a*l2_c - l1_c*l2_a)/ l2_a * l1_B
        x = ((-1)*line2.c)/line2.a
        y = (line1_a*line2.c - line1_c*line2.a)/(line2.a * line1_b)
        return (x,y)
    elif line1.isVertical:
        line2_a, line2_b, line2_c = convertToNormalForm(line2)

        # x = (-l1_c)/l1_a; y = (l2_a*l1_c - l2_c*l1_a)/ l1_a * l2_B
        x = ((-1)*line2.c)/line2.a
        y = (line2_a*line1.c - line2_c*line1.a)/(line1.a * line2_b)
        return (x,y)
    else:
        x = (line2.yintercept - line1.yintercept)/(line1.slope - line2.slope)
        y = line1.slope*x + line1.yintercept

        return (x,y)

def perpendicularIntersectionPointToLine(pt: shpPoint, lineEq: LineEquation):
    if lineEq.isHorizontal:
        # this means that our slope is 0, so we can't compute the slope will be UNDEFINED
        # Solution: since the line is horizontal, pt.x stays the same, and y is replaced by line's y-intercept
        return (pt.x, lineEq.yintercept)
    if lineEq.isVertical:
        return None

    perpLine_slope = (-1)/lineEq.slope
    perpLine_yintercept = pt.y - perpLine_slope*pt.x

    perpLine = LineEquation(perpLine_slope, perpLine_yintercept)

    return intersectionPoint(perpLine, lineEq)

# def perpendicularIntersectionPointToLine(pt: shpPoint, lineEq: LineEquation):
#     perpLine_slope = (-1)/lineEq.slope
#     perpLine_yintercept = pt.y - perpLine_slope*pt.x

#     perpLine = LineEquation(perpLine_slope, perpLine_yintercept)

#     return intersectionPoint(perpLine, lineEq)

def convertSegListToSimplGeomLS(segList: list, ptsList: list, edgeSimplif: Edge):
    # Generate a new LineString from newSegmentList
        newPtsList = []
        for segIdx in range(0,len(segList)):
            seg: Segment = segList[segIdx]
            if segIdx == len(segList)-1: 
                #if we have the last element, add both the end point and the first points
                newPtsList.append(ptsList[seg.startId])
                newPtsList.append(ptsList[seg.endId])
            else:
                newPtsList.append(ptsList[seg.startId])

        simplifiedLS: shpLS = shpLS(newPtsList)

        if edgeSimplif.id == 8750:
            plotShpLS(simplifiedLS, "green")

        newGeom = convertPtListToSimplGeomLS(newPtsList)
        newEdge = Edge(
            edgeSimplif.id,
            edgeSimplif.start_node_id,
            angle(newGeom[0],newGeom[1]),
            edgeSimplif.end_node_id,
            angle(newGeom[-1], newGeom[-2]),
            edgeSimplif.left_face_id,
            edgeSimplif.right_face_id,
            newGeom,
            {} #no info for now
        )
        return newEdge

def convertSegListToSimplGeomOnly(segList: list, ptsList: list):
    # Generate a new LineString from newSegmentList
        newPtsList = []
        for segIdx in range(0,len(segList)):
            seg: Segment = segList[segIdx]
            if segIdx == len(segList)-1: 
                #if we have the last element, add both the end point and the first points
                newPtsList.append(ptsList[seg.startId])
                newPtsList.append(ptsList[seg.endId])
            else:
                newPtsList.append(ptsList[seg.startId])

        simplifiedLS: shpLS = shpLS(newPtsList)

        #ShpLS(simplifiedLS, "green")

        return convertPtListToSimplGeomLS(newPtsList)

