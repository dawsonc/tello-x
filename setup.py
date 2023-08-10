"""
Sample project setup file.
"""
from setuptools import setup, find_packages


setup(
    name="sample-project-MichaelCurrin",
    version="1.1.0", # Change to 0.0.1 for new projects.
    author="Michael Currin",
    author_email="",
    description="A sample Python package project",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    license="MIT",
    url="https://github.com/MichaelCurrin/python-package-quickstart",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
