import tests.matrix_tests.matrix_benchmark as benchmark
import tests.matrix_tests.matrix_test as tester


def print_name_and_run(name, func, *args, **kwargs):
    print(name + ": start")
    result = func(*args, **kwargs)
    print(name + ": done!")
    return result


def compare_and_benchmark(MatrixA, MatrixB, do_compare=True, do_benchmark=True):
    test_name = f"{MatrixA.__name__} vs {MatrixB.__name__}"
    print(test_name)
    if do_compare:
        print_name_and_run(test_name + '-compare', tester.subtest_compare, MatrixA, MatrixB)
    if do_benchmark:
        print_name_and_run(test_name + '-benchmark', benchmark.subtest_bench, MatrixA, MatrixB)
    print(test_name + ": done!")

