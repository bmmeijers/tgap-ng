#!/usr/bin/env python

from tgap_ng.tgap import main
from tgap_ng.datastructure import retrieve
from tgap_ng.edge_simplify.samsonov_yakimova2 import simplifySY

from shapely import wkt, errors as shpErr

DATASET, unbounded_id = "top10nl_limburg_tiny", 0
SRID = 28992
BASE_DENOMINATOR = 10000

if __name__ == '__main__':
    print('USE THIS FILE TO TEST INDIVIDUAL LINES FOR SIMPLIFICATION')
    

    pp = retrieve(DATASET, SRID, unbounded_id)

    shpGeomDict: dict = {}
    for edgeId in pp.edges:
        try:
            shpGeom = wkt.loads(pp.edges[edgeId].geometry.wkt)
            #print(shpGeom)

            shpGeomDict[edgeId] = shpGeom

        except shpErr.WKTReadingError as err:
            print(f"Error while transforming the geom.wkt to shp LineString: {err}")

    edge5285 = pp.edges[5285]
    edge5303 = pp.edges[5303] #closed, highly angular structure
    edge8757 = pp.edges[8757]

    simplifySY(edge5303, pp, 0, gpdGeom= shpGeomDict)
