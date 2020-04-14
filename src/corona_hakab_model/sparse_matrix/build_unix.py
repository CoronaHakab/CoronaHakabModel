import platform
import subprocess
from pathlib import Path
from sysconfig import get_paths

import numpy as np
from swimport import ContainerSwim, FileSource, Function, Swim, Typedef, pools

COMPILE_ADDITIONAL_INCLUDE_DIRS = [
    get_paths()['include'],
    get_paths()['platinclude'],
    np.get_include(),
]
COMPILE_ADDITIONAL_INCLUDE_LIBS = [
    get_paths()['stdlib'],
    get_paths()['platstdlib'],
    get_paths()['platlib'],
    get_paths()['purelib'],
]


def write_swim():
    src = FileSource("sparse.hpp")
    swim = Swim("sparse")

    swim.add_raw("%nodefaultctor ManifestMatrix;")
    swim.add_raw("%nodefaultctor SparseMatrix;")
    swim.add_python_begin(
        """
    # flake8: noqa
    from sparse_base import SparseBase, ManifestBase

    import numpy as np
    size_t = np.dtype('uint64')
    """
    )

    swim(pools.include(src))

    swim(pools.primitive(additionals=False, out_iterable_types=()))
    swim(pools.list("size_t"))
    swim(pools.list("std::vector<size_t>"))

    swim(Typedef.Behaviour()(src))

    swim(pools.numpy_arrays(typedefs=tuple({"size_t": "unsigned long", "dtype": "float"}.items())))
    swim(pools.tuple("dtype", "dtype"))
    swim(pools.print())

    oswim = ContainerSwim("MagicOperator", src, director=True)
    oswim(Function.Behaviour())

    mswim = ContainerSwim("ManifestMatrix", src, wrapper_superclass='"ManifestBase"')
    mswim(Function.Behaviour())
    mswim.extend_py_def(
        "I_POA",
        "self, v, op",
        """
        nz = np.flatnonzero(v).astype(np.uint64, copy=False)
        return self.I_POA.prev(self, v, nz, op)
        """,
    )
    mswim.extend_py_def(
        "__getitem__",
        "self,item",
        """
        i, j = item
        return self.get(i, j)
        """,
    )

    pswim = ContainerSwim("SparseMatrix", src, )
    pswim(Function.Behaviour())
    pswim.extend_py_def(
        "__getitem__",
        "self,item",
        """
        i, j = item
        if not self.has_value(i, j):
            return None
        return self.get(i, j)
        """,
    )
    pswim.extend_py_def(
        "manifest",
        "self, sample=None, global_prob_factor=1",
        """
        if sample is None:
            sample = generator.random(self.nz_count(), dtype=np.float32)
        if global_prob_factor != 1:
            sample /= global_prob_factor
        return self.manifest.prev(self, sample)
        """,
    )
    pswim.extend_py_def(
        "batch_set",
        "self,row,columns,probs,values",
        """
        columns = np.asanyarray(columns, dtype=size_t)
        probs = np.asanyarray(probs, dtype=np.float32)
        values = np.asanyarray(values, dtype=np.float32)
        return self.batch_set.prev(self, row,columns,probs,values)
        """,
    )

    swim(oswim)
    swim(mswim)
    swim(pswim)

    swim.write("sparse.i")


def run_swim():
    interface_file = "sparse.i"
    subprocess.run(["swig", "-c++", "-python", "-py3", interface_file], stdout=None, check=True)


def compile():
    optimization = "-O2"  # in case of fire, set to Od

    src_path = Path("sparse.cpp")
    cxx_path = "../sparse_matrix/sparse_wrap.cxx"

    proc = subprocess.run(
        ["g++", optimization, "-fPIC", "-std=c++17", "-c", str(src_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if proc.returncode != 0:
        print(proc.stdout)
        raise Exception(f"cl returned {proc.returncode}")

    proc = subprocess.run(
        [
            "g++",
            optimization,
            "-fPIC",
            "-std=c++17",
            "-Wno-error=narrowing",
            "-c",
            str(cxx_path),
            *[f"-I{inc}" for inc in COMPILE_ADDITIONAL_INCLUDE_DIRS],
            *[f"-L{inc}" for inc in COMPILE_ADDITIONAL_INCLUDE_LIBS],
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if proc.returncode != 0:
        print(proc.stdout)
        raise Exception(f"cl returned {proc.returncode}")

    if platform.system() == "Darwin":  # MacOS
        cmd = ["ld", "-bundle", "-flat_namespace", "-undefined", "suppress"]
    elif platform.system() == "Linux":
        cmd = ["g++", "-shared"]
    else:
        raise OSError("only support Linux and MacOS!")

    proc = subprocess.run(
        [*cmd, "-o", "_sparse.so", cxx_path.split(".cxx")[0] + ".o", str(src_path).split(".")[0] + ".o"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if proc.returncode != 0:
        print(proc.stdout)
        raise Exception(f"cl returned {proc.returncode}")


def rm_aux_files():
    files_to_rm_prefix = "sparse"
    files_to_rm_extensions = ["o", "i", "cxx"]

    cwd = Path(".")
    for p in sum([list(cwd.glob(f"{files_to_rm_prefix}*.{ext}")) for ext in files_to_rm_extensions], []):
        p.unlink()


def build():
    write_swim()
    run_swim()
    compile()
    rm_aux_files()


if __name__ == "__main__":
    build()
