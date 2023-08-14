# Contributing to `tellox`

First off: thanks for your interest in contributing! We'd love to hear your ideas for improvement!

This document goes over some best practices for contributing to this project.

## Installing the development environment

You can install the development environment using [`poetry`](https://python-poetry.org/docs/) with

```
poetry install --with dev
poetry run pre-commit install
```

## Code standards

Please make sure that all of the following tools run and pass before contributing a PR.

```
poetry run black .   # auto-format
poetry run isort .   # sort imports
poetry run flake8 .  # check format
poetry run mypy .    # static type checking
poetry run pytest .  # tests
```

All of these (except `pytest`) are also run as pre-commit hooks.
