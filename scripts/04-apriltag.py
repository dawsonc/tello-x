"""An example of using tellox to detect apriltags."""
from time import sleep

import tellox as tx

# Make a pilot object. Make a window to visualize the camera feed.
pilot = tx.Pilot()

# New images are only displayed when requested, so we need to request them
framerate = 10.0  # Hz
while True:
    img = pilot.get_camera_frame(visualize=False)

    # Detect apriltags in the image
    tags = pilot.detect_tags(img, visualize=True)

    sleep(1 / framerate)
