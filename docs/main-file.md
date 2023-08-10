# Main file

If you have a structure like this:

```
foo/
    __main__.py
```

Then you can run:

```sh
$ python -m foo
```

Even though `foo` is a directory, it can be run as a module, using the `__main__.py` entry point.
