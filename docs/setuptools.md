# Setuptools
> Intro to using `setuptools` library for packaging a project


## Setup file

Sample from the tutorial:

- `setup.py`
    ```python
    import setuptools


    with open("README.md", "r") as fh:
        long_description = fh.read()

    setuptools.setup(
        name="example-pkg-YOUR-USERNAME-HERE",
        version="0.0.1",
        author="Example Author",
        author_email="author@example.com",
        description="A small example package",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://github.com/pypa/sampleproject",
        packages=setuptools.find_packages(),
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
        ],
        python_requires='>=3.6',
    )
    ```

Notes:

- The `name` field might be more flexible on case and style but it was recommended as `example-pkg-YOUR-USERNAME-HERE`, 
- Projects mostly handle the long description in a single like as `long_description=open("README.md").read()`. This is still readable and safe (as the script finishes quickly and so no file descriptor is left open).
- Some metadata fields like `"long_description_content_type"` will only work on newer versions `setuptools>=38.6.0` - see [issue](https://github.com/di/markdown-description-example/issues/4). Update `sdist` as well, to avoid warnings.
- There's a warning that if author name is setup, then author email must be set too.
- Another approach is using `distutils` but I don't know which approach is better.
    ```python
    from distutils.core import setup
    ```
- Instead of using `find_packages`, you can also just set:
    ```python
    packages=['foo']
    ```


## Install and upgrade system dependencies

Make sure you have the latest version of `setuptools` installed at the user level for Python 3. i.e. Run this _outside_ a virtual environment and with the `--user` flag.

From the tutorial:

```sh
$ python3 -m pip install --user --upgrade setuptools wheel
```

If you don't use the `--user` flag, you might get errors on install for Python managed by root and then have to use `sudo`, which should be avoided in general when installing Python packages.

Those are stored in site packages. e.g. `.local/lib/python3.5/site-packages` on Linux.

Follow the [Installation](installation.md) page for installing the project package.
