from util import dist, lower_bound, upper_bound


def test_ubound():
    assert (
            upper_bound(dist(10))
            == upper_bound(dist(5, 10))
            == upper_bound(dist(2, 3, 10))
            == 10
    )


def test_lbound():
    assert (
            lower_bound(dist(-10))
            == lower_bound(dist(-10, 5))
            == lower_bound(dist(-10, 3, 6))
            == -10
    )