from operator import index
from geopandas import GeoSeries, geoseries

from .datastructure import retrieve

from shapely import errors as shErr
from shapely.geometry import LineString

import matplotlib

DATASET, unbounded_id = "top10nl_limburg_tiny", 0
SRID = 28992
BASE_DENOMINATOR = 10000

def testGeoSeriesIntersection():
    #test to see how intersections can be detected using GeoSeries
    ls1 = LineString([(1.0, 1.0), (3.0, 3.0)])
    ls2 = LineString([(1.0, 3.0), (3.0, 1.0)])
    ls3 = LineString([(1.0, 1.0), (1.0, 3.0)])
    ls4 = LineString([(1.0, 3.0), (3.0, 3.0)])

    #initialize the geoseries of both intersecting and non-intersecting pairs of lines
    geom1 = GeoSeries([ls1])
    geom2 = GeoSeries([ls2])
    geom3 = GeoSeries([ls3])
    geom4 = GeoSeries([ls4])

    geom1.plot()
    geom2.plot()
    geom3.plot()
    geom4.plot()


def testGeopandasStrucutre():
#Test to see how Shapely LineString / geopandas work
    geoSrs: GeoSeries = GeoSeries()

    pp = retrieve(DATASET, SRID, unbounded_id)
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

def main():
    #testGeopandasStrucutre()
    testGeoSeriesIntersection()

    print("got here")


if __name__ == "__main__":
    main()