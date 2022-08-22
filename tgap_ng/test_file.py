from operator import index
from geopandas import GeoSeries, geoseries

#from .datastructure import retrieve

from shapely import errors as shErr
from shapely.geometry import LineString

import matplotlib.pyplot as plt

from quadtree import QuadTree

DATASET, unbounded_id = "top10nl_limburg_tiny", 0
SRID = 28992
BASE_DENOMINATOR = 10000

def testGeoSeriesIntersection():
    # test to see how intersections can be detected using GeoSeries
    ls1 = LineString([(1.0, 1.0), (3.0, 3.0)])
    ls2 = LineString([(1.0, 3.0), (3.0, 1.0)])
    ls3 = LineString([(1.0, 1.0), (1.0, 3.0)])
    ls4 = LineString([(3.0, 1.0), (3.0, 3.0)])

    # Compare line pairs. The result should be as follows:
    # l1 x l2 -> intersects (true)
    # l3 x l4 -> not (false)
    # l1 x l3 (or l2 x l4, ...) -> one common point -> ?
    print(f"l1 x l2 -> {ls1.intersection(ls2)}")
    print(f"l3 x l4 -> {ls3.intersection(ls4)}")
    print(f"l1 x l3 (common starting point)-> {ls1.intersection(ls3)}")

    #NOTE: Findings (good to remember!) .intersection returns the point(s) where the two linestrings intersect.

    print("Stop Debug")

    # initialize the geoseries of both intersecting and non-intersecting pairs of lines
    # NOTE: GeoSeries seems to be good when it comes to grouping together multiple polylines
    # Might be useful in our case, by grouping together all polylines/edges, maybe it will be easier for our operations.
    # However, this might mean changing the implementation (i.e. how edges are treated in our PlanarPartition)
    # geom1 = GeoSeries([ls1])
    # geom2 = GeoSeries([ls2])
    # geom3 = GeoSeries([ls3])
    # geom4 = GeoSeries([ls4])

    # Plot to see the lines (separate plots since they're different geoseries)
    # geom1.plot()
    # geom2.plot()
    # geom3.plot()
    # geom4.plot()

    # plt.show()

    # Comparing the GeoSeries
    # res = geom1.intersects(geom2.geometry)
    # print(f"l1 x l2 -> {res}")
    # print(f"l3 x l4 -> {geom3.intersects(geom4.geometry)}")
    # print(f"l1 x l3 -> {geom1.intersects(geom3.geometry)}")


def testGeopandasStrucutre():
#Test to see how Shapely LineString / geopandas work
    geoSrs: GeoSeries = GeoSeries()

    pp = None #TODO FIX THIS#retrieve(DATASET, SRID, unbounded_id)
    test_wkt_str = ["LineString (190946.701 308956.517, 190938.974 308937.474,"+
                    "190924.951 308909.693, 190902.461 308874.504, 190881.823 308847.516,"+
                    "190867.007 308830.583, 190837.109 308806.77, 190806.412 308783.159)"]
    
    lineTest = LineString([(190946.701, 308956.517), (190938.974, 308937.474), 
                (190924.951, 308909.69), (190902.461, 308874.504), (190881.823, 308847.516), 
                (190867.007, 308830.583), (190837.109, 308806.77)])

    print("original:",lineTest.wkt)
    #convert geometry to GeoSeries
    for edgeId in pp.edges:
        print(pp.edges[edgeId].geometry)

        try:
            wktGeom = pp.edges[edgeId].geometry.wkt
            print("Old wkt:", wktGeom)
            
            #transform wkt to have a space between LineString and ()
            indexParanth = wktGeom.find('(')
            #wktGeomAdapted = wktGeom[:indexParanth] + " " + wktGeom[indexParanth:]
            #wktGeomAdapted = ["LineString " + wktGeom[indexParanth:]]
            #print("New wkt:", wktGeomAdapted)


            newGeom = GeoSeries.from_wkt([wktGeom]) #NOTE: An array needs to be given as input!

            pp.edges[edgeId].geom = newGeom
        except shErr.WKTReadingError as e:
            print(e)

def testQuadTree():
    qt = QuadTree(
        [
            (-100, -100),
            (100, 100),
        ],
        64,
    )

    pt1 = (6,6)
    qt.add(pt1)

    minXY = (3,3)

    maxXY = (9,9)
    
    qt.range_search((minXY, maxXY))
    print(qt)

def main():
    """
    I use this script to test out how different libraries such as Shapely and GeoPandas work. 
    """
    #testGeopandasStrucutre()
    testGeoSeriesIntersection()
    #testQuadTree()

    print("Ended execution of selected test functions")


if __name__ == "__main__":
    main()