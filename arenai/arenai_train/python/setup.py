from setuptools import find_packages, setup

setup(
    name="arenai_executorch_conversion",
    version="1.0.0",
    author="Samuel Berrien",
    packages=find_packages(
        include=[
            "arenai_executorch_conversion",
            "arenai_executorch_conversion.*",
        ]
    ),
)
