from time import time


def timeit_print(func):
    def timed(*args, **kw):
        ts = time()
        result = func(*args, **kw)
        te = time()
        if 'log_time' in kw:
            name = kw.get('log_name', func.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r  %2.2f ms' % \
                  (func.__name__, (te - ts) * 1000))
        return result
    return timed
