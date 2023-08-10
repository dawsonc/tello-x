# Installation


## Install system dependencies

Install Python 3.6 or higher.

## Setup virtual environment

Create a virtual environment.

Activate it.


## Install project dependencies 

Several approaches are covered below. Choose one.

### Install directly from project

This requires this project to be cloned, or downloaded and unzipped from GitHub

Then install the project:

```sh
$ python setup.py sdist bdist_wheel
```

<!-- 

TBC if these from SoloLearn are still accurate as is differs from the above.

Build a source distribution:

```sh
$ python setup.py sdist
```

Build a binary distribution:

```sh
$ python setup.py bdist
# Windows
> python setup.py bdist_wininst
```

-->

### Install from packages repository

- [PyPI](https://pypi.org/) - only for a published package.
    ```python
    $ pip install PACKAGE_NAME
    ```
- GitHub - the repo just needs to be public.
    ```sh
    $ pip install -e git+https://github.com/MichaelCurrin/python-package-quickstart.git@master
    ```

Change the reference at the end with a commit reference like a commit hash, branch name or tag.

### Install from Python egg

Based on [blog post](https://adamj.eu/tech/2019/03/11/pip-install-from-a-git-repository/).

> Any URL may use the #egg=name syntax (see VCS Support) to explicitly state the project name. [source](https://pip.pypa.io/en/stable/reference/pip_install/)

```
[-e] git+https://git.example.com/MyProject#egg=MyProject
```

Tip: That line can be added directly to your requirements file, if you want to install a GitHub repo as a dependency of your main project.

Examples of installing using pip.

- Git repo
    ```sh
    $ pip install -e 'git+https://github.com/django/django.git@45dfb3641aa4d9828a7c5448d11aa67c7cbd7966#egg=django[argon2]'
    ```
- Tarball
    ```sh
    $ pip install -e 'https://github.com/django/django/archive/45dfb3641aa4d9828a7c5448d11aa67c7cbd7966.tar.gz#egg=django[argon2]'
    ```

Remember to use quotes so that characters like hash are escaped.
The dependencies can be included in hard brackets. I don't know tbe point if this because that can get covered during installation as normal, using setup or requirements file.

### Install from zipped archive

This uses a tarball which GitHub creates for you at a commit reference. You do not have to make it.

#### By file path

Download a tarball file and install from the file path.

```sh
$ pip install master.tar.gz
```

#### By URL

Install directly from the repo:

e.g.

- GitHub
    ```sh
    $ pip install https://github.com/django/django/archive/master.tar.gz
    ```
- GitLab
    ```sh
    $ pip install https://gitlab.com/pycqa/flake8/-/archive/3.7.7/flake8-3.7.7.tar.gz
    ```
- Bitbucket
    ```sh
    $ pip install https://bitbucket.org/hpk42/tox/get/2.3.1.tar.gz
    ```
