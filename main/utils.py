import time


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        print('Function: %r took %.2f sec' % (method.__name__, (te - ts)))
        return result

    return timed
