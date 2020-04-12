import itertools as it
import subprocess
from pathlib import Path

from swimport import ContainerSwim, FileSource, Function, Swim, Typedef, pools

PY_ROOT = r"T:\py_envs\3.8"
SWIG_PATH = r"T:\programs\swigwin-4.0.1\swig.exe"
PY_INCLUDE_PATH = PY_ROOT + r"\include"
PY_LIB_PATH = PY_ROOT + r"\libs\python38.lib"

windows_kit_template = r"C:\Program Files (x86)\Windows Kits\10\{}\10.0.17763.0" + "\\"
MSVC_dir = r"C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.24.28314" + "\\"
np_include_path = PY_ROOT + r"\Lib\site-packages\numpy\core\include\\"
windows_kit_include = windows_kit_template.format("include")
windows_kit_lib = windows_kit_template.format("lib")

CL_PATH = MSVC_dir + r"bin\Hostx64\x64\cl.exe"
COMPILE_ADDITIONAL_INCLUDE_DIRS = [
    MSVC_dir + "include",
    windows_kit_include + "ucrt",
    windows_kit_include + "shared",
    windows_kit_include + "um",
    np_include_path,
]
COMPILE_ADDITIONAL_LIBS = [
    MSVC_dir + r"lib\x64\libcpmt.lib",
    MSVC_dir + r"lib\x64\libcmt.lib",
    MSVC_dir + r"lib\x64\oldnames.lib",
    MSVC_dir + r"lib\x64\libvcruntime.lib",
    MSVC_dir + r"lib\x64\libconcrt.lib",
    windows_kit_lib + r"um\x64\kernel32.lib",
    windows_kit_lib + r"ucrt\x64\libucrt.lib",
    windows_kit_lib + r"um\x64\Uuid.lib",
]

optimization = "/O2"  # in case of fire, set to Od


def write_swim():
    src = FileSource("parasymbolic.hpp")
    swim = Swim("parasymbolic")

    swim.add_raw("%nodefaultctor;")
    swim.add_python_begin("import numpy as np")
    swim.add_python_begin("from contextlib import contextmanager")

    swim(pools.include(src))

    swim(pools.primitive(additionals=False, out_iterable_types=()))
    swim(pools.list("size_t"))
    swim(pools.list("std::vector<size_t>"))
    swim(pools.list("std::vector<std::vector<size_t>>"))

    swim(Typedef.Behaviour()(src))

    swim(pools.numpy_arrays(typedefs=tuple({"size_t": "unsigned long long", "dtype": "float"}.items())))
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
    subprocess.run(
        [SWIG_PATH, "-c++", "-python", "-py3", "parasymbolic.i"], stdout=None, check=True  # '-debug-tmsearch',
    )


def compile():
    tmpdir = Path("temp")
    src_path = Path("parasymbolic.cpp")
    cxx_path = "parasymbolic_wrap.cxx"

    tmpdir.mkdir(exist_ok=True, parents=True)
    # todo compile with warnings?
    proc = subprocess.run(
        [
            CL_PATH,
            "/nologo",
            "/LD",
            "/EHsc",
            "/utf-8",
            optimization,
            "/Tp",
            str(cxx_path),
            *(("/Tp", str(src_path)) if src_path.exists() else ()),
            "/Fo:" + str(tmpdir) + "\\",
            "/I",
            PY_INCLUDE_PATH,
            *it.chain.from_iterable(("/I", i) for i in COMPILE_ADDITIONAL_INCLUDE_DIRS),
            "/link",
            "/LIBPATH",
            PY_LIB_PATH,
            *it.chain.from_iterable(("/LIBPATH", l) for l in COMPILE_ADDITIONAL_LIBS),
            "/IMPLIB:" + str(tmpdir / "example.lib"),
            "/OUT:" + "_parasymbolic.pyd",
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    if proc.returncode != 0:
        print(proc.stdout)
        raise Exception(f"cl returned {proc.returncode}")


def build():
    write_swim()
    run_swim()
    compile()


if __name__ == "__main__":
    build()
