# Module initialization
> About the `__init__.py` file

See the [Packages](https://docs.python.org/3/tutorial/modules.html#packages) section of official tutorial.


## Purpose

What is an `__init__.py` file for?

> The `__init__.py` files are required to make Python treat directories containing the file as packages. This prevents directories with a common name, such as string, unintentionally hiding valid modules that occur later on the module search path.
>
> In the simplest case, `__init__.py` can just be an empty file, but it can also execute initialization code for the package or set the `__all__` variable, described later.


## Package installation

Reduce to the fewest pieces, an installable Python package is a directory with an `__init__.py` file. This file could be empty and do nothing. For a simple project of a single script, that file could contain all the Python code and no other files are needed.


## Adding a docstring

Adding a docstring to the otherwise empty file can be useful when reading your documentation (especially through the console) to explain a module and its purpose. e.g.

- `foo/bar/__init__.py`
    ```python
    """
    Foo initialization module.
    """

    ```

An initialization file will be run when its parent directory is imported.


## Adding global code

This can be useful to run side effects such as a print statement which is not inside a function.

- `foo/bar/__init__.py`
    ```python
    """
    Foo initialization module.
    """

    print("This is the foo module")
    ```

Add a nested module.

- `foo/bar/__init__.py`
    ```python
    """
    Bar initialization module.
    """

    print("This is the bar module")
    ```


Then import:

```
>>> import foo
This is the foo module
```

```
>>> import foo.bar
This is the foo module
This is the bar module
```


## Adding functions

More commonly, any functions which are shared for the directory can be placed in the `__init__.py` file.

- `foo/__init__.py`
    ```python
    """
    Foo initialization module.
    """

    def hello():
        print("Hello, world!")


    def main():
        print("This is main function")
        hello()

    if __name__ == '__main__':
        main()
    ```

Example use:

```
>>> import foo
>>> foo.hello()
Hello, world!
```

```
$ python -m foo
This is main function
Hello, world!
```

Or

```
$ python -m foo.__init__
This is main function
Hello, world!
```

This will work too in this simple case. But it will execute from inside the `foo` directory rather than the top-level directory, so some imports within the script or references to data files might fail.

```sh
$ python foo/__init__.py
This is main function
Hello, world!
```

## Is it needed?

From [StackOverflow answer](https://stackoverflow.com/questions/37139786/is-init-py-not-required-for-packages-in-python-3-3)

Python 3.3+ supports *Implicit Namespace Packages* - which means you can have a package without an `__init__.py` file.

```
parent_package/
    __init__.py      # EMPTY, NOT NECESSARY in Python 3.3+
    child_package/
        __init__.py  # STILL REQUIRED if you want to run an initialization script
        child1.py
        child2.py
        child3.py
```
