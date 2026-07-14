from __future__ import annotations

import json
from datetime import datetime

from agent.config import get_config


def send_robot_message(
    message,
    *,
    request_image: bool = False,
    topic: str = "/eye_in_hand_cam_rgb",
) -> None:
    """Send a message to the robot over WebSocket (synchronous)."""
    from websockets.sync.client import connect

    cfg = get_config()
    payload = {
        "message": message,
        "timestamp": str(datetime.now()),
        "request_image": request_image,
        "topic": topic,
    }

    with connect(cfg.robot.websocket_uri) as websocket:
        print("[robot] connected")
        websocket.send(json.dumps(payload))
        print("[robot] payload sent")

        if request_image:
            response = websocket.recv()
            if isinstance(response, bytes):
                image_path = cfg.root / cfg.vision.image_path
                image_path.write_bytes(response)
                print(f"[robot] image saved: {image_path}")
