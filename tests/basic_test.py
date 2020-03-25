from util import upper_bound, dist, lower_bound


def basic_test():
    pass


def ubound_test():
    assert upper_bound(dist(10)) == upper_bound(dist(5, 10)) == upper_bound(dist(2, 3, 10)) == 10


def lbound_test():
    assert lower_bound(dist(-10)) == lower_bound(dist(-10, 5)) == lower_bound(dist(-10, 3, 6)) == -10
