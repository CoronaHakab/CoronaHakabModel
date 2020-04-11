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
    src = FileSource("parasymbolic.hpp")
    swim = Swim("parasymbolic")

    swim.add_raw("%nodefaultctor;")
    swim.add_python_begin("import numpy as np")
    swim.add_python_begin("from contextlib import contextmanager")

    swim(pools.include(src))

    swim(pools.primitive())

    swim(Typedef.Behaviour()(src))

    swim(pools.numpy_arrays(typedefs=tuple({"size_t": "unsigned long", "dtype": "float"}.items())))
    pswim = ContainerSwim("ParasymbolicMatrix", src)

    pswim(r"operator\*=" >> Function.Behaviour(append_python="return self"))
    pswim(Function.Behaviour())
    pswim.extend_py_def(
        "prob_any",
        "self, v",
        """
                        nz = np.flatnonzero(v).astype(np.uint64, copy=False)
                        return self._prob_any(v, nz)
                        """,
    )
    pswim.extend_py_def(
        "__setitem__",
        "self, key, v",
        """
                        comp, row, indices = key
                        indices = np.asanyarray(indices, dtype=np.uint64)
                        v = np.asanyarray(v, dtype=np.float32)
                        self.batch_set(comp, row, indices, v)
                        """,
    )
    pswim.extend_py_def(
        "lock_rebuild",
        "self",
        """
                        self.set_calc_lock(True)
                        yield self
                        self.set_calc_lock(False)
                        """,
        wrapper="contextmanager",
    )
    swim(pswim)

    swim.write("parasymbolic.i")


def run_swim():
    wrapper_file = "parasymbolic_wrap.cxx"
    interface_file = "parasymbolic.i"
    subprocess.run(["swig", "-c++", "-python", "-py3", "-o", wrapper_file, interface_file], stdout=None, check=True)


def compile():
    src_path = "parasymbolic.cpp"
    cxx_path = "parasymbolic_wrap.cxx"
    optimization = "-O2"  # in case of fire, set to Od

    proc = subprocess.run(
        ["g++", optimization, "-fPIC", "-std=c++17", "-c", src_path],
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
            cxx_path,
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
        [*cmd, "-o", "_parasymbolic.so", cxx_path.split(".")[0] + ".o", src_path.split(".")[0] + ".o"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if proc.returncode != 0:
        print(proc.stdout)
        raise Exception(f"cl returned {proc.returncode}")


def rm_aux_files():
    files_to_rm_prefix = "parasymbolic"
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
