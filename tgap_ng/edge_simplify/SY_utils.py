from __future__ import annotations

from math import sqrt

from .SY_constants import Direction

from shapely.geometry import Point as shpPoint, LineString as shpLS

from tgap_ng.datastructure import Edge

import matplotlib.pyplot as plt

import random

from quadtree import QuadTree

def decideBias() -> Direction:
    # Randomly decide between Left and Right directional bias
    return random.choice(list(Direction))

def createPoint(x,y) -> shpPoint:
    return shpPoint(x,y)

def computeLength(pt1, pt2):
    return sqrt(pow(pt2[0]-pt1[0],2) + pow(pt2[1]-pt1[1],2))


def convertSimplPtsToShp(ptList) -> list[shpPoint]:
        #get a list of coordinates like (x y) and convert them to Shapely Point
        shpPtList = []
        for pt in ptList:
            shpPtList.append(createPoint(pt[0],pt[1]))

        return shpPtList

def plotShpLS(line: shpLS, color: str):
    x,y = line.xy
    plt.plot(x,y,c=color,marker='o')
    plt.show()

def safelyAppendToDict(dictInstace: dict[list], anyKey, anyObject):
    # append to list if key exists, or create list then append to it
    if anyKey in dictInstace:
        dictInstace[anyKey].append(anyObject)
    else:
        dictInstace[anyKey] = [anyObject]

def safelyAddShpPtToQuadTree(qt: QuadTree, shpPt: shpPoint) -> bool:
    # checks if point already exists in QuadTree. If it does, return False,
    # otherwise add it and return True
    pt = (shpPt.x, shpPt.y)
    if qt.__contains__(pt):
        #print("Point already exists in QuadTree")
        return False
    qt.add(pt)
    return True

def safelyRemoveShpPtFromQuadTree(qt: QuadTree, shpPt: shpPoint) -> bool:
    # if QT contains point, then reomove it (and return True for successful operation). Else, return False
    pt = (shpPt.x, shpPt.y)
    if qt.__contains__(pt):
        qt.remove(pt)
        return True
    return False

def quadTreeRangeSearchAdapter(ptsList: list[shpPoint], qt: QuadTree):
    # Works as an adapter over the original range_search in QuadTree
    # Get a list of points and compute minX, minY, maxX, maxY from them
    # then change them in a format that is accepted by the quad tree
    minX = 100000000
    minY = 100000000
    maxX = -100000000
    maxY = -100000000

    for pt in ptsList:
        if pt.x < minX:
            minX = pt.x
        elif pt.x > maxX:
            maxX = pt.x
        elif pt.y < minY:
            minY = pt.y
        elif pt.y > maxY:
            maxY = pt.y

    maxXY = (maxX, maxY)
    minXY = (minX, minY)

    return qt.range_search((minXY, maxXY))