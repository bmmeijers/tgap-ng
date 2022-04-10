#!/usr/bin/env python

from tgap_ng.tgap import main
from tgap_ng.datastructure import retrieve
#from tgap_ng.edge_simplify.samsonov_yakimova2 import simplifySY
from tgap_ng.edge_simplify.SYSimplified import simplifySYSimple as simplifySY

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
    edge8801 = pp.edges[8801] #open, simplify first segment
    edge9333 = pp.edges[9333]

    simplify5303 = False
    simplify8801 = False
    simplify9333 = True

    if simplify5303:
        simpl1 = simplifySY(edge5303, pp, 0, gpdGeom= shpGeomDict)
        shpGeomDict[5303] = wkt.loads(simpl1.geometry.wkt) #change the geometry in shpGeomDict

        simpl2 = simplifySY(simpl1, pp, 0, gpdGeom= shpGeomDict)
        shpGeomDict[5303] = wkt.loads(simpl2.geometry.wkt) 

        simpl3 = simplifySY(simpl2, pp, 0, gpdGeom= shpGeomDict)
        shpGeomDict[5303] = wkt.loads(simpl3.geometry.wkt)

        simpl4 = simplifySY(simpl3, pp, 0, gpdGeom= shpGeomDict)
        shpGeomDict[5303] = wkt.loads(simpl4.geometry.wkt)

        simpl5 = simplifySY(simpl4, pp, 0, gpdGeom= shpGeomDict)
        shpGeomDict[5303] = wkt.loads(simpl5.geometry.wkt)

        simpl6 = simplifySY(simpl5, pp, 0, gpdGeom= shpGeomDict)
        shpGeomDict[5303] = wkt.loads(simpl6.geometry.wkt)

    if simplify8801:
        simpl1 = simplifySY(edge8801, pp, 0)

        simpl2 = simplifySY(simpl1, pp, 0)

        simpl3 = simplifySY(simpl2, pp, 0)

        simpl4 = simplifySY(simpl3, pp, 0)

    if simplify9333:
        simpl1 = simplifySY(edge9333, pp, 0)

        simpl2 = simplifySY(simpl1, pp, 0)

        simpl3 = simplifySY(simpl2, pp, 0)

        simpl4 = simplifySY(simpl3, pp, 0)