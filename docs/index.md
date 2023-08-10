# Python Packages Tutorial 🐍 📦
>  A simple template for a Python project that can be installed with pip - includes docs and tutorial
  
Go to source code:

[![MichaelCurrin - python-package-quickstart](https://img.shields.io/static/v1?label=MichaelCurrin&message=python-package-quickstart&color=blue&logo=github)](https://github.com/MichaelCurrin/python-package-quickstart)


## About

This project is a quick reference, tutorial and demo for making a Python project installable using the `pip install PACKAGE` command. For full details, see the linked resources.

- _Python 3.6+_
- _Project settings configured for VS Code_


## Resources

### Official docs

- Python Packaging site
    - [packaging.python.org](https://packaging.python.org) Homepage
    - [Packaging Python Projects](https://packaging.python.org/tutorials/packaging-projects/) tutorial. Most of this guide is based on that.
    - [Installing packages](https://packaging.python.org/tutorials/installing-packages/) tutorial.
    - Key projects
        - [twine](https://packaging.python.org/key_projects/#twine) package - for upload to PyPI.
        - [setuptools](https://packaging.python.org/key_projects/#setuptools) package.
        - [wheel](https://packaging.python.org/key_projects/#wheel) package.
- Python 3 docs
    - [Modules](https://docs.python.org/3/tutorial/modules.html#packages) tutorial.
- PyPI site
    - [pypi.org](https://pypi.org/) homepage.
    - [Help](https://pypi.org/help/) page. See Basics for answers:
        - What's a package, project, or release?
        - How do I install a file (package) from PyPI?
        - How do I package and publish my code for PyPI?

### Other

- [How to Publish Your Own Python Package to PyPI](https://realpython.com/courses/how-to-publish-your-own-python-package-pypi/)


## Source Distributions vs Wheels

From [pip docs](https://packaging.python.org/tutorials/installing-packages/#source-distributions-vs-wheels)

> `pip` can install from either Source Distributions (sdist) or Wheels, but if both are present on PyPI, pip will prefer a compatible wheel.
>
> Wheels are a pre-built distribution format that provides faster installation compared to Source Distributions (sdist), especially when a project contains compiled extensions.
>
> If `pip` does not find a wheel to install, it will locally build a wheel and cache it for future installs, instead of rebuilding the source distribution in the future.


## Related projects

- [MichaelCurrin/py-project-template](https://github.com/MichaelCurrin/py-project-template)


## Guide

### What you'll get at the end of this tutorial

- A standard well-structured Python project outline.
- Your project can be installed using `pip` or `python setup.py`.
- Make your project installable using `pip` and a Github URL. 

Note on publishing: 

- This does not cover publishing to PyPI, as that is more restrictive especially if you're a beginner for packaging. For PyPI, you have to setup an account, meet the quality standards (maybe get approval?) and your package becomes searchable. 
- A Github package is lower profile and has lower barriers to experiment with. Also, you can make your project private for your user or org, but still install it where you have access.

### Typical structure

Here the repo or cloned folder is `sample-project` with a hyphen.

Inside the repo is `sampleproject` which may not have a hyphen (otherwise you can't import it as package) and using an underscore is discouraged. See [PEP-8](https://www.python.org/dev/peps/pep-0008/#id40).

```
sample-project
  sampleproject/
    foo/
      __init__.py
    __init__.py
    sampleproject.py
  tests/
  setup.py
  LICENSE
  README.md
```

Note that a directory and a script are both considered a "module" in Python, if it can be imported.

### Init file

See [init file](init-file.md) tutorial page.

### Main file

See [main file](main-file.md) tutorial page.

A main file is a script or an entrypoint to be run directly, but it might be considered a module if you import it from elsewhere. A convention is that this main script matches package name as in the example. You can also make it `__main__.py`.

### Intra-package references

See [section](https://docs.python.org/3/tutorial/modules.html#intra-package-references) of tutorial.

```python
from . import echo
from .. import formats
from ..filters import equalizer
```

This does not work in the main module.
