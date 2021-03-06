import setuptools

from __data__ import __author__, __author_email__, __license__, __url__, __version__

setuptools.setup(
    name="corona_hakab_model",
    version=__version__,
    author=__author__,
    author_email=__author_email__,
    license=__license__,
    url=__url__,
    description="Corona Hakab Model Simulator",
    package_dir={"": "src"},
    packages=["corona_hakab_model", "corona_hakab_model_data"],
    install_requires=["numpy", "scipy", "scikit-learn", "matplotlib", "pytest", "pandas"],
    extras_require={
        "quality": ["pytest", "tox", "flake8", "isort", "black"],
        "profiling": ["yappi"],
        "pretty graphs": ["pyside2"],
        "building": ["swimport"],
    },
    python_requires=">=3.8.0",
    include_package_data=True,
    data_files=[("", ["README.md", "LICENSE"])],
)
