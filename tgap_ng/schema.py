import sys
import logging
logging.basicConfig(
    format='%(asctime)s %(levelname)s (%(process)d) %(message)s', 
    stream=sys.stderr, 
    level=logging.DEBUG)
from collections import namedtuple
from sink import Field, Layer, Schema, Index

def output_layers(name, srid):
    """tables/layers that can be saved in the database as output"""
#     tgap = namedtuple('tGAP', 
#                       ('face', 'face_geometry', 'face_hierarchy', 'edge', 'node'))
    class tGAP(object):
        def __init__(self):
            self.face = None
#            self.face_geometry = None
            self.face_hierarchy = None
            self.edge = None
            self.node = None
            self.edge_statistics = None
        def __iter__(self):
            return iter([self.face, 
                         #self.face_geometry, 
                         self.face_hierarchy, self.edge, self.node, self.edge_stats])

    tgap = tGAP()
    # imp values
    #imp_low = Field("imp_low", "float8")
    #imp_high = Field("imp_high", "float8")
    #imp_own = Field("imp_own", "float8")
    # step
    step_low = Field("step_low", "integer") # max: 2147483647
    step_high = Field("step_high", "integer")

    gstep_low = Field("gstep_low", "integer") # max: 2147483647
    gstep_high = Field("gstep_high", "integer")

    # face
    face_id = Field("face_id", "integer")
    area = Field("area", "numeric")
    klass = Field("feature_class", "integer")
    
    ###### DEVELOPMENT GROUPS ########
#     groups1 = Field("groups1", "boolean")
#     groups2 = Field("groups2", "boolean")
#     groups3 = Field("groups3", "boolean")
#     groups4 = Field("groups4", "boolean")
    ###########
    mbr = Field("mbr_geometry", "box2d") 
    pip = Field("pip_geometry", "point")
    parent_id = Field("parent_face_id", "integer")
    polygon = Field("geometry", "polygon")
    # edge
    edge_id = Field("edge_id", "integer")
    left_low = Field("left_face_id_low", "integer")
    right_low = Field("right_face_id_low", "integer")
    left_high = Field("left_face_id_high", "integer")
    right_high = Field("right_face_id_high", "integer")
    start = Field("start_node_id", "integer")
    end = Field("end_node_id", "integer")
    edge_class = Field("edge_class", "integer") # why not also name this feature_class ???
    pickled_blg = Field("pickled_blg", "varchar")
    smoothline = Field("smooth", "json")
    path = Field("geometry", "linestring", dimension = 1) # FIXME: dimension -> needed?
    # node
    node_id = Field("node_id", "integer")
    coord = Field("geometry", "point", dimension = 0)     # FIXME: dimension -> needed?
    #
    face_schema = Schema(
        [face_id, 
         #imp_low, imp_high, 
         step_low, step_high,
         #gstep_low, gstep_high,
         #imp_own, 
         area, 
         klass,
         ###### DEVELOPMENT GROUPS ########
#          groups1,
#          groups2,
#          groups3,
#          groups4, 
         ###### DEVELOPMENT GROUPS ########
         mbr, pip],
        [Index(fields = [face_id, step_low], primary_key = True), 
         #Index(fields = [imp_low]),
         #Index(fields = [imp_high]),
         Index(fields = [mbr])
         ]
    )
#     face_geometry_schema = Schema(
#         [face_id,
#          imp_low, imp_high,
#          step_low, step_high, 
#          polygon 
#          ],
#         [Index(fields = [face_id, imp_low], primary_key = True), 
#          Index(fields = [polygon])
#          ]
#     )
    #
    face_hier_schema = Schema(
        [face_id,
         # imp_low, imp_high,
         step_low, step_high,
         parent_id,
         ],
        [Index(fields = [face_id, step_low], primary_key = True),
         #Index(fields = [imp_low]),
         #Index(fields = [imp_high]),
         Index(fields = [parent_id]), ]
    )
    #
    edge_schema = Schema(
        [edge_id, 
         step_low, step_high, 
         start, end, 
         left_low, right_low, left_high, right_high, 
         #imp_low, imp_high,

         #gstep_low, gstep_high,
         edge_class,
         #pickled_blg,
         #smoothline,
         path],
        [Index(fields = [edge_id, step_low], primary_key = True),
         #Index(fields = [imp_low]),
         #Index(fields = [imp_high]),
         Index(fields = [step_low]),
         Index(fields = [step_high]),
         Index(fields = [path])]
    )
    #
    node_schema = Schema(
        [node_id, coord],
        [Index(fields = [node_id]),#, primary_key = True),
         Index(fields = [coord])]
    )

    face_step = Field("face_step", "integer")
    edge_ct_simplified = Field("edges_simplified", "integer") # max: 2147483647
    edge_ct_total = Field("edges_total", "integer")

    edge_stats_schema = Schema(
        [face_step, edge_ct_simplified, edge_ct_total],
        []
    )


    #
    #folder = '/tmp/' + name + '/'
    #if not os.path.exists(folder):
    #    os.mkdir(folder)
#    prefix = 'tgap_'
#    face_file = tempfile.NamedTemporaryFile(suffix='_face', 
#        prefix=prefix, 
#        #dir=folder, 
#        delete = False
#    )
##     face_geometry_file = tempfile.NamedTemporaryFile(suffix='_face_geometry', 
##         prefix=prefix, 
##         #dir=folder, 
##         delete = False
##     )
#    face_hier_file = tempfile.NamedTemporaryFile(suffix='_face_hier', 
#        prefix=prefix, 
#        #dir=folder, 
#        delete = False
#    )
#    edge_file = tempfile.NamedTemporaryFile(suffix='_edge', 
#        prefix=prefix, 
#        #dir=folder, 
#        delete = False
#    )
#    node_file = tempfile.NamedTemporaryFile(suffix='_node', 
#        prefix=prefix, 
#        #dir=folder, 
#        delete = False
#    )
    face_layer = Layer(face_schema, 
                                "{0}_tgap_face".format(name), 
                                srid = srid)
#     face_geometry_layer = StreamingLayer(face_geometry_schema, 
#                                 "{0}_tgap_face_geometry".format(name), 
#                                 srid = srid, 
#                                 stream=face_geometry_file)
    face_hier_layer = Layer(face_hier_schema, 
                            "{0}_tgap_face_hierarchy".format(name), 
                            srid = srid)
    edge_layer = Layer(edge_schema, 
                                "{0}_tgap_edge".format(name), 
                                srid = srid)
    node_layer = Layer(node_schema, 
                                "{0}_tgap_node".format(name), 
                                srid = srid)

    edge_stats_layer = Layer(edge_stats_schema, 
                                "{0}_tgap_edge_stats".format(name), 
                                srid = srid)

    tgap.face = face_layer
#     tgap.face_geometry = face_geometry_layer
    tgap.face_hierarchy = face_hier_layer
    tgap.edge = edge_layer
    tgap.edge_stats = edge_stats_layer
    tgap.node = node_layer
    return tgap


if __name__ == "__main__":
    from .loader import AsyncLoader

    # define output object
    output = output_layers('test_mm', 28992)

    # add some output to one of the tables
#    output.face_hierarchy.append(1, 0, 0, 0, 0, 10)

    # load the layers of the tables to the database

    # initialize tables
    with AsyncLoader(workers=1) as loader:
        for table in output:
            loader.load_schema(table)
        del table

    # load data
    with AsyncLoader(workers=4) as loader:
        for table in output:
            loader.load_data(table)
        del table

    # finalize tables: indexing + clustering
    with AsyncLoader(workers=4) as loader:
        for table in output:
            loader.load_indexes(table)
        del table

    # finalize tables: statistics
    with AsyncLoader(workers=4) as loader:
        for table in output:
            loader.load_statistics(table)
        del table

#        output.face_hierarchy.clear()

