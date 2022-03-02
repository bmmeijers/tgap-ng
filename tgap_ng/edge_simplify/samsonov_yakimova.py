from tgap_ng.datastructure import PlanarPartition

def simplifySY(edgeToBeSimplified, pp: PlanarPartition, tolerance, DEBUG = False):
    """
    Method used for simplifying a polyline having characteristics of man-made structures 
    (i.e. orthogonal turns inbetween segments)

    Simplification based on the paper "Shape-Adaptive Geometric Simplification of Heterogeneous Line Datasets"
    by Samsonov and Yakimova. Further info on this algorithm can be found in 3.1.1. Simpl. of orthogonal segments.

    Topological awareness also needs to be accounted for, both in the case of self-intersections (which are mentioned
    in the paper) as well as with neighbouring structures (which is beyond the scope of that paper).
    """
    print("Entered SY Simplification Module")

    for line in edgeToBeSimplified.geometry:
        print("Lines:" ,line)

    #convert geometry.wkt to Shapely LineString object
    wktGeom= edgeToBeSimplified.geometry.wkt



    return [None, None]

def segmentLength():
    pass