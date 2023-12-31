"""An example of using tellox to get video from the drone."""
from time import sleep

import tellox as tx

# Make a pilot object. Make a window to visualize the camera feed.
pilot = tx.Pilot(visualize=True)

# New images are only displayed when requested, so we need to request them
framerate = 10.0  # Hz
while True:
    pilot.get_camera_frame()
    sleep(1 / framerate)
