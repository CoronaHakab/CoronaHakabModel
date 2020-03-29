import subprocess

from swimport import FileSource, Swim, pools, ContainerSwim, Function

PY_ROOT = r"C:\python_envs\x64\3.8"
SWIG_PATH = r"T:\programs\swigwin-4.0.1\swig.exe"
PY_INCLUDE_PATH = PY_ROOT + r'\include'
PY_LIB_PATH = PY_ROOT + r'\libs\python37.lib'

windows_kit_template = r'C:\Program Files (x86)\Windows Kits\10\{}\10.0.17763.0' + "\\"
MSVC_dir = r'C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Tools\MSVC\14.16.27023' + "\\"
np_include_path = PY_ROOT + r"\Lib\site-packages\numpy\core\include\\"
windows_kit_include = windows_kit_template.format('include')
windows_kit_lib = windows_kit_template.format('lib')

CL_PATH = MSVC_dir + r'bin\Hostx64\x64\cl.exe'
COMPILE_ADDITIONAL_INCLUDE_DIRS = [
    MSVC_dir + 'include',
    windows_kit_include + 'ucrt',
    windows_kit_include + 'shared',
    windows_kit_include + 'um',
    np_include_path
]
COMPILE_ADDITIONAL_LIBS = [
    MSVC_dir + r'lib\x64\libcpmt.lib',
    MSVC_dir + r'lib\x64\libcmt.lib',
    MSVC_dir + r'lib\x64\oldnames.lib',
    MSVC_dir + r'lib\x64\libvcruntime.lib',
    windows_kit_lib + r'um\x64\kernel32.lib',
    windows_kit_lib + r'ucrt\x64\libucrt.lib',
    windows_kit_lib + r'um\x64\Uuid.lib'
]

optimization = '/O2'


def write_swim():
    src = FileSource("parasymbolic.hpp")
    swim = Swim("parasymbolic")

    swim(
        pools.include(src)
    )

    bswim = ContainerSwim("BareSparseMatrix", src)
    cswim = ContainerSwim("CoffedSparseMatrix", src)
    pswim = ContainerSwim("ParasymbolicMatrix", src)

    bswim(Function.Behaviour())
    cswim(Function.Behaviour())
    pswim(Function.Behaviour())

    swim(bswim)
    swim(cswim)
    swim(pswim)

    swim.write("parasymbolic.i")


def run_swim():
    subprocess.run([SWIG_PATH, '-c++', '-python', '-py3',  # '-debug-tmsearch',
                    "parasymbolic.i"],
                   stdout=None, check=True)


def compile(): pass


if __name__ == '__main__':
    write_swim()
    run_swim()
