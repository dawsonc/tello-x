# Tello-X 🚁
> The easy way to control a DJI Tello drone from Python.

[![GitHub tag](https://img.shields.io/github/tag/dawsonc/tello-x?include_prereleases=&sort=semver)](https://github.com/dawsonc/tello-x/releases/)
[![License](https://img.shields.io/badge/License-MIT-blue)](#license)

## Features

- Easily pilot the Tello drone by sending commands and reading sensor data.
- Built-in support for [AprilTag](https://github.com/duckietown/lib-dt-apriltags) detection.
- Built-in data logging

## Installation

`pip install tello-x`

That's it! Connect to your Tello over WiFi and try some of the [examples](https://github.com/dawsonc/tello-x/scripts).

## A note on coordinate frames

The Tello's body frame is defined North-East-Down (a standard for aircraft), so `x` points forward (out of the camera), `y` points to the right of the drone, and `z` points down.
The camera frame is defined with `z` pointing out of the camera and `x` pointing to the right of the drone. If you use `pilot.detect_tags` to get the relative pose of AprilTags in the drone's view, then the returned translation `pose_t` and rotation `pose_R` are defined *relative to the camera frame*. To convert the tag position to the drone body frame, use `pilot.convert_to_drone_frame`.

## License

Released under [MIT](/LICENSE) by [Charles Dawson](https://github.com/dawsonc).
