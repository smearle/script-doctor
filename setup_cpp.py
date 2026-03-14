import os
import sys
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

try:
    import pybind11
except ModuleNotFoundError:
    pybind11 = None


def _get_pybind11_include() -> str:
    include_override = os.environ.get('PYBIND11_INCLUDE_DIR')
    if include_override:
        include_path = Path(include_override).expanduser()
        if include_path.is_dir():
            return str(include_path)
        raise FileNotFoundError(
            f"PYBIND11_INCLUDE_DIR points to a missing directory: {include_path}"
        )

    if pybind11 is not None:
        return pybind11.get_include()

    repo_root = Path(__file__).resolve().parent
    for venv_name in ('.venv', '.venv-jaxtar', '.venv-py312'):
        for include_path in (repo_root / venv_name).glob('lib/python*/site-packages/pybind11/include'):
            if include_path.is_dir():
                return str(include_path)

    raise ModuleNotFoundError(
        "pybind11 headers were not found. Install pybind11 in the active environment "
        "or set PYBIND11_INCLUDE_DIR to a pybind11/include directory."
    )


class BuildExt(build_ext):
    def build_extensions(self):
        if sys.platform == 'darwin':
            for ext in self.extensions:
                ext.extra_compile_args.append('-mmacosx-version-min=10.14')
                ext.extra_link_args.append('-mmacosx-version-min=10.14')
        if sys.platform.startswith('linux'):
            use_openmp = os.environ.get('PUZZLESCRIPT_CPP_OPENMP', '1') != '0'
            if use_openmp:
                for ext in self.extensions:
                    ext.extra_compile_args.append('-fopenmp')
                    ext.extra_link_args.append('-fopenmp')
        build_ext.build_extensions(self)


ext_modules = [
    Extension(
        'puzzlescript_cpp._puzzlescript_cpp',
        sources=[
            'puzzlescript_cpp/src/engine.cpp',
            'puzzlescript_cpp/src/heuristic.cpp',
            'puzzlescript_cpp/src/batched_engine.cpp',
            'puzzlescript_cpp/src/renderer.cpp',
            'puzzlescript_cpp/src/solver.cpp',
            'puzzlescript_cpp/src/bindings.cpp',
        ],
        include_dirs=[
            _get_pybind11_include(),
            'puzzlescript_cpp/src',
        ],
        language='c++',
        extra_compile_args=['-std=c++17', '-O2', '-Wall'],
    ),
]

setup(
    name='puzzlescript_cpp',
    version='0.1.0',
    description='C++ PuzzleScript engine with pybind11 bindings',
    packages=['puzzlescript_cpp'],
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExt},
    python_requires='>=3.8',
    install_requires=[
        'pybind11>=2.10',
        'numpy',
    ],
)
