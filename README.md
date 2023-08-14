# Tello-X 🚁
> The easy way to control a DJI Tello drone from Python.

[![GitHub tag](https://img.shields.io/github/tag/dawsonc/tello-x?include_prereleases=&sort=semver)](https://github.com/dawsonc/tello-x/releases/)
[![License](https://img.shields.io/badge/License-MIT-blue)](#license)
[![PyPI version](https://badge.fury.io/py/tellox.svg)](https://badge.fury.io/py/tellox)

## Features

- Easily pilot the Tello drone by sending commands and reading sensor data.
- Built-in support for [AprilTag](https://github.com/pupil-labs/apriltags) detection.
- Built-in data logging

## Installation

We recommen installing `tellox` in a virtual environment:
```
# Create a virtual environment for your project
cd path/to/your/project
python -m venv venv

# Activate the virtual environment (you'll need to do this whenever you start a new shell)
source venv/bin/activate  # or venv/Scripts/activate in Windows PowerShell

# Install this library
pip install tellox
```

That's it! Connect to your Tello over WiFi and try some of the [examples](https://github.com/dawsonc/tello-x/scripts).

For a minimal example:
```
import time
import tellox as tx

pilot = tx.Pilot()
pilot.takeoff()
pilot.land()
```

## A note on coordinate frames

The Tello's body frame is defined North-East-Down (a standard for aircraft), so `x` points forward (out of the camera), `y` points to the right of the drone, and `z` points down.
The camera frame is defined with `z` pointing out of the camera and `x` pointing to the right of the drone. If you use `pilot.detect_tags` to get the relative pose of AprilTags in the drone's view, then the returned translation `pose_t` and rotation `pose_R` are defined *relative to the camera frame*. To convert the tag position to the drone body frame, use `pilot.convert_to_drone_frame`.

## License

Released under the [MIT License](/LICENSE) by [Charles Dawson](https://github.com/dawsonc).

## Contributing

If you find a bug or would like to contribute a feature, that's awesome! Please file an [issue](https://github.com/dawsonc/tello-x/issues) and open a [pull request](https://github.com/dawsonc/tello-x/issues). More details on contribution can be found in the [`CONTRIBUTING.md`](https://github.com/dawsonc/tello-x/blob/master/CONTRIBUTING.md) file.
