"""
Setup file for tello-x.
"""
from setuptools import setup

setup(
    name="tellox",
    version="0.0.1",
    author="Charles Dawson",
    author_email="cbd@mit.edu",
    description="The easy way to control a DJI Tello drone.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    license="MIT",
    url="https://github.com/dawsonc/tello-x",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: System :: Hardware",
        "Framework :: Robot Framework :: Library",
    ],
    packages=["tellox"],
    install_requires=[
        "numpy",
        "opencv-python",
        "pandas",
        "djitellopy",
        "dt-apriltags",
        "matplotlib",
    ],
    python_requires=">=3.6",
)
