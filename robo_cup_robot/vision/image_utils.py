"""Convert ROS Image messages to NumPy arrays without cv_bridge binaries."""

import numpy as np


def _image_rows(message, bytes_per_pixel):
    row_bytes = message.width * bytes_per_pixel
    if message.step < row_bytes:
        raise ValueError(
            f"Image step {message.step} is smaller than row size {row_bytes}"
        )

    raw = np.frombuffer(message.data, dtype=np.uint8)
    expected_size = message.height * message.step
    if raw.size < expected_size:
        raise ValueError(
            f"Image data has {raw.size} bytes; expected at least {expected_size}"
        )
    rows = raw[:expected_size].reshape(message.height, message.step)
    return np.ascontiguousarray(rows[:, :row_bytes])


def image_msg_to_numpy(message, desired_encoding):
    """Decode common RealSense color and depth encodings into NumPy arrays."""
    encoding = message.encoding.lower()

    if desired_encoding == "bgr8":
        if encoding in ("bgr8", "8uc3"):
            return _image_rows(message, 3).reshape(
                message.height, message.width, 3
            )
        if encoding == "rgb8":
            rgb = _image_rows(message, 3).reshape(
                message.height, message.width, 3
            )
            return np.ascontiguousarray(rgb[:, :, ::-1])
        if encoding in ("bgra8", "rgba8"):
            rgba = _image_rows(message, 4).reshape(
                message.height, message.width, 4
            )
            color = rgba[:, :, :3]
            if encoding == "rgba8":
                color = color[:, :, ::-1]
            return np.ascontiguousarray(color)

    if desired_encoding == "16UC1" and encoding in ("16uc1", "mono16"):
        rows = _image_rows(message, 2)
        byte_order = ">" if message.is_bigendian else "<"
        wire_dtype = np.dtype(f"{byte_order}u2")
        return rows.view(wire_dtype).reshape(message.height, message.width).astype(
            np.uint16, copy=False
        )

    raise ValueError(
        f"Cannot convert ROS image encoding {message.encoding!r} to {desired_encoding!r}"
    )
