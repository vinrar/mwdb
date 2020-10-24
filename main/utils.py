import pprint
import time


# Add @timeit on top of your function for printing time taken in sec by your function
# @timeit
# def your_function()
def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        print('Function: %r took %.2f sec' % (method.__name__, (te - ts)))
        return result

    return timed


def print_dict(dictionary):
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(dictionary)
