import multiprocessing
try:
    from io import StringIO
except ImportError:
    from io import StringIO
from contextlib import closing
import logging

import connection
from sink.backends.postgis import dump_schema, dump_indices, dump_statistics
from sink.backends.postgis_copy import dump_data



class CopyFromMessage(object):
    def __init__(self, table, columns, data):
        self.table = table
        self.columns = columns
        self.data = data


# Modes for postgres cursors:
# ---------------------------
#0 -> autocommit
#1 -> read committed
#2 -> serialized (but not officially supported by pg)
#3 -> serialized


class ReadCommittedMessage(object):
    # -- will be passed to cursor with isolation level=1,
    # i.e. within a normal database transaction
    def __init__(self, payload):
        self.payload = payload


class AutoCommitMessage(object):
# -- will be passed to cursor with isolation level=0 (e.g. for vacuum), 
# i.e. no transaction will be used
    def __init__(self, payload):
        self.payload = payload


# TODO 
# We should handle errors in the pattern: e.g. use a result queue that we
# check, and in case an error occurs for loading the data or making the table
# or ..., stop the whole procedure!

# -- the worker function that performs the load to the database
def worker(queue):
    """Worker that performs the work in the database

    Work can be specified by one of the following message types:

    - Load data (CopyFromMessage)
    - Perform maintenance tasks (AutoCommitMessage), isolation level = 0
    - Make Schema / Indexes (ReadCommitedMessage), isolation level = 1

    """
    import timeit
    # db = connection.connection(geo_enabled = True)
    # pattern here is as follows:
    # --
    # for i, job in enumerate(iter(queue.get, None)):
    #    # execute the calculation
    #    output_queue.put(result), note - no output queue is used
    # shutdown the worker process
    # --
    for i, msg in enumerate(iter(queue.get, None), start=1):
        logging.debug("{} {} {}".format(multiprocessing.current_process().name, "handles", id(msg)))
        start = timeit.default_timer()
        # the postgres server process shows a steady increase 
        # in its memory usage if we leave open the connection
        # so we re-open the connection for every message we handle
        # to prevent high memory usage also at dbms server side
        with connection.connection(True) as db:
            if isinstance(msg, ReadCommittedMessage):
#                with closing(StringIO(msg.payload)) as stream:
#                    db.execute(stream.getvalue())
                db.execute(msg.payload)
            elif isinstance(msg, AutoCommitMessage):
#                with closing(StringIO(msg.payload)) as stream:
                db.execute(msg.payload, isolation_level=0)
            elif isinstance(msg, CopyFromMessage):
                with closing(StringIO(msg.data)) as stream:
                    db.copy_from(stream, msg.table, columns=msg.columns, sep="\t") 
        now = timeit.default_timer()
        logging.debug("{} {} {} {} {} {} {}".format(multiprocessing.current_process().name, "handled", id(msg), type(msg), "taking", "{:.4f}".format(now - start), "sec(s)"))
    logging.debug("{} {}".format(multiprocessing.current_process().name, "closed"))


class BaseLoader(object):
    """Load data to Postgres"""
    def __init__(self, workers=multiprocessing.cpu_count(), high_water_mark=None):
        if high_water_mark is None:
            high_water_mark = int(1.5 * workers)
        self.queue = multiprocessing.Queue(high_water_mark)
        self.procs = []
        for i in range(workers):
            p = multiprocessing.Process(name="Worker #{0}".format(i), 
                                        target=worker,
                                        args=(self.queue,))
            self.procs.append(p)
            p.start()

    def __enter__(self):
        return self

    def __exit__(self, a, b, c):
        return self.done()

    def load(self, item):
        self.queue.put(item)

    def done(self):
        # -- put poison pill 'None' in the queue to close workers
        for i in range(len(self.procs)):
            self.queue.put(None)
        # -- wait for all workers to finish their task
        try:
            self.queue.close()
            self.queue.join_thread()
            for p in self.procs:
                p.join()
        except KeyboardInterrupt:
            logging.debug('parent received ctrl-c')
            for p in self.procs:
                p.terminate()
                p.join()


class SinkLoaderMixin(object):
    """Mixin for methods targetted at loading Sink layers"""
    def load_schema(self, layer):
        # -- make schema
        with closing(StringIO()) as stream:
            dump_schema(layer, stream)
            s = stream.getvalue()
            self.load(ReadCommittedMessage(s))

    def load_data(self, layer):
        # -- load data
        with closing(StringIO()) as stream:
            dump_data(layer, stream)
            self.load(CopyFromMessage(layer.name, 
                                      layer.schema.names,
                                      stream.getvalue()))

    def load_indexes(self, layer):
        # -- make the indexes
        with closing(StringIO()) as stream:
            dump_indices(layer, stream)
            self.load(ReadCommittedMessage(stream.getvalue()))

    def load_statistics(self, layer):
        # -- make the statistics
        with closing(StringIO()) as stream:
            dump_statistics(layer, stream)
            self.load(AutoCommitMessage(stream.getvalue()))


class AsyncLoader(SinkLoaderMixin, BaseLoader):
    """Asynchronous database loader"""
    pass


def example():
    from sink import Field, Schema, Index, Layer

    from simplegeom.geometry import Point
    from random import randint

    import timeit

    # -- let's define the structure for a little table to hold points
    SRID = 28992
    gid = Field("gid", "numeric")
    geo_point = Field("geo_point", "point")
    i_gid_pkey = Index([gid], primary_key = True)
    i_geo_point = Index([geo_point], cluster = True)
    schema = Schema([gid, geo_point], [i_gid_pkey, i_geo_point])
    layer = Layer(schema, "test_table", SRID)

    # -- make the schema
    with AsyncLoader(workers=1) as loader:
        loader.load_schema(layer)

    # -- load the data in chunks
    with AsyncLoader() as loader:
        prev = timeit.default_timer()
        for i in range(1, int(5e5)):
            layer.append(i, Point(randint(0, 1e8), randint(0, 1e6), SRID))
            # check every ten points, whether the layer has more than 50000 features
            # if so, dump the data and clear the layer
            if (i % 10000 == 0) and len(layer) >= 50000:
                now = timeit.default_timer()
#                print now - prev, len(layer)
                prev = now
                loader.load_data(layer)
                layer.clear()
        # in case of remaining data, also load this data
        if len(layer) > 0:
            loader.load_data(layer)
            layer.clear()

    with AsyncLoader(workers=1) as loader:
        # -- indexes
        loader.load_indexes(layer)
        # -- statistics
        loader.load_statistics(layer)


if __name__ == "__main__":
    example()

