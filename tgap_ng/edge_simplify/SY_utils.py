from math import sqrt
from .SY_DataStrucuctures import LineEquation, Segment

from shapely.geometry import Point as shpPoint, LineString as shpLS

from tgap_ng.datastructure import Edge, angle

import matplotlib.pyplot as plt

from simplegeom import geometry as simplgeom

from __future__ import annotations

def createPoint(x,y) -> shpPoint:
    return shpPoint(x,y)

def computeLength(pt1, pt2):
    return sqrt(pow(pt2[0]-pt1[0],2) + pow(pt2[1]-pt1[1],2))

def intersectionPoint(line1: LineEquation, line2: LineEquation):
    # considering we have two line equations: y = m1*x + b1 and y = m2*x + b2
    # X_intersection = (b2 -b1)/(m1-m2) and Y_intersection = (m1*xintersection) + b2
    # extdStartSeg - Line 1 ; perpSeg - Line 2    
    x = (line2.yintercept - line1.yintercept)/(line1.slope - line2.slope)
    y = line1.slope*x + line1.yintercept

    return (x,y)

def perpendicularIntersectionPointToLine(pt: shpPoint, lineEq: LineEquation):
    perpLine_slope = (-1)/lineEq.slope
    perpLine_yintercept = pt.y - perpLine_slope*pt.x

    perpLine = LineEquation(perpLine_slope, perpLine_yintercept)

    return intersectionPoint(perpLine, lineEq)

def convertSimplPtsToShp(ptList) -> list[shpPoint]:
        #get a list of coordinates like (x y) and convert them to Shapely Point
        shpPtList = []
        for pt in ptList:
            shpPtList.append(createPoint(pt[0],pt[1]))

        return shpPtList

def printSmallestSegment(self):
    print(f"Smallest segment has id {self.smlstSegId}, len: {self.smlstSegLen}")

def convertPtListToSimplGeomLS(ptList: list):
    # Convert our segments to LineString of type SimpleGeometry
    orderedPtList = []
    for pt in ptList:
        simplePt = simplgeom.Point(pt[0], pt[1], 28992)
        orderedPtList.append(simplePt)

    return simplgeom.LineString(orderedPtList)

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

def plotShpLS(line: shpLS, color: str):
    x,y = line.xy
    plt.plot(x,y,c=color,marker='o')
    plt.show()
