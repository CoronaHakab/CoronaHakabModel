import itertools as it
import subprocess
import sys
from pathlib import Path

from json import load
from typing import Dict

from swimport import ContainerSwim, FileSource, Function, Swim, Typedef, pools

PY_ROOT = Path(sys.executable).parent
PY_INCLUDE_PATH = PY_ROOT / "include"
PY_LIB_PATH = PY_ROOT / r"libs\python38.lib"

np_include_path = PY_ROOT / r"Lib\site-packages\numpy\core\include\\"

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


def run_swim(config):
    subprocess.run(
        [config['SWIG_PATH'], "-c++", "-python", "-py3", "parasymbolic.i"], stdout=None, check=True
        # '-debug-tmsearch',
    )


def compile(config: dict):
    tmpdir = Path("temp")
    src_path = Path("parasymbolic.cpp")
    cxx_path = "parasymbolic_wrap.cxx"

    tmpdir.mkdir(exist_ok=True, parents=True)

    # todo compile with warnings?
    proc = subprocess.run(
        [
            config["CL_PATH"],
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
            *it.chain.from_iterable(("/I", i) for i in config["COMPILE_ADDITIONAL_INCLUDE_DIRS"]),
            "/link",
            "/LIBPATH",
            PY_LIB_PATH,
            *it.chain.from_iterable(("/LIBPATH", l) for l in config["COMPILE_ADDITIONAL_LIBS"]),
            "/IMPLIB:" + str(tmpdir / "example.lib"),
            "/OUT:" + "_parasymbolic.pyd",
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    if proc.returncode != 0:
        print(proc.stdout)
        raise Exception(f"cl returned {proc.returncode}")


def update_config(config):
    config["CL_PATH"] = str(Path(config["MSVC_DIR"]) / r"bin\Hostx64\x64\cl.exe")
    windows_kit_include = Path(config["WINDOWS_KIT_DIR"]) / \
                          "include" / \
                          config["WINDOWS_KIT_VERSION"]
    windows_kit_lib = Path(config["WINDOWS_KIT_DIR"]) / \
                      "lib" / \
                      config["WINDOWS_KIT_VERSION"]
    config["COMPILE_ADDITIONAL_INCLUDE_DIRS"] = [
        Path(config["MSVC_DIR"]) / "include",
        windows_kit_include / "ucrt",
        windows_kit_include / "shared",
        windows_kit_include / "um",
        np_include_path,
    ]
    config["COMPILE_ADDITIONAL_LIBS"] = [
        Path(config["MSVC_DIR"]) / r"lib\x64\libcpmt.lib",
        Path(config["MSVC_DIR"]) / r"lib\x64\libcmt.lib",
        Path(config["MSVC_DIR"]) / r"lib\x64\oldnames.lib",
        Path(config["MSVC_DIR"]) / r"lib\x64\libvcruntime.lib",
        Path(config["MSVC_DIR"]) / r"lib\x64\libconcrt.lib",
        Path(windows_kit_lib) / r"um\x64\kernel32.lib",
        Path(windows_kit_lib) / r"ucrt\x64\libucrt.lib",
        Path(windows_kit_lib) / r"um\x64\Uuid.lib",
    ]


def build(config: Dict):
    write_swim()
    update_config(config)
    run_swim(config)
    compile(config)


if __name__ == "__main__":
    build_inputs = sys.argv[1:]
    if len(build_inputs) < 1:
        build_config_file = str(Path(__file__).parent / "build_config.json")
    else:
        build_config_file = build_inputs[0]
    with open(build_config_file, 'r') as config_fh:
        build_config_file = load(config_fh)
    build(build_config_file)
