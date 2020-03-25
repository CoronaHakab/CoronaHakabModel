from scipy.stats import binom, randint, rv_discrete


def dist(*args):
    def const_dist(a):
        return rv_discrete(name="const", values=([a], [1]))()

    def uniform_dist(a, b):
        return randint(a, b + 1)

    def trig(a, c, b):
        # todo I have no idea what this distribution supposedly represents, we're gonna pretend it's
        #  an offset-binomial and call it a day

        return binom(b - a, (c - a) / (b - a), loc=a)

    if len(args) == 1:
        return const_dist(*args)
    if len(args) == 2:
        return uniform_dist(*args)
    if len(args) == 3:
        return trig(*args)
    raise TypeError


def upper_bound(d):
    return d.b + d.kwds.get("loc", 0)


def lower_bound(d):
    return d.a + d.kwds.get("loc", 0)
