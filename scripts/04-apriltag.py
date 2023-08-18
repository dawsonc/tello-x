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

    # Get the euler angles of the detected tags
    if tags:
        for tag in tags:
            position, _, euler_angles = pilot.get_drone_pose_in_tag_frame(tag)
            print("Tag ID: {}".format(tag.tag_id))
            print("Drone position in tag frame: {}".format(position))
            print("Roll, pitch, yaw: {}".format(euler_angles))

    sleep(1 / framerate)
