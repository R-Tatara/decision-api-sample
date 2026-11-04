import base64
import sys
from datetime import datetime

import cv2
import numpy as np
import pyrealsense2 as rs
from openai import OpenAI

WIDTH = 640
HEIGHT = 480
FPS = 30
PROBABILITY_FONT_SCALE = 0.8
PROBABILITY_MARGIN = 12
PROBABILITY_LINE_GAP = 6


def setup_realsense() -> rs.pipeline:
    """Start the RealSense color stream and return the running pipeline."""
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
    pipeline.start(config)
    return pipeline


def get_frame(pipeline: rs.pipeline) -> np.ndarray | None:
    """Block until the next color frame arrives and return it."""
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    if not color_frame:
        return None
    image = np.asanyarray(color_frame.get_data())
    return image


def classify_image(image: np.ndarray, client: OpenAI) -> list[tuple[str, float]] | None:
    """Classify a captured image using the decision API."""
    encoded, image_buffer = cv2.imencode(".png", image)
    if not encoded:
        raise ValueError("Failed to encode frame as PNG")
    image_b64 = base64.b64encode(image_buffer.tobytes()).decode("ascii")

    labels = ["closed", "opened", "unclear"]

    decision = client.decisions.create(
        model="gpt-6-luna",
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": "Look at the transparent lid on the black box. Is the lid open or closed?",
                    },
                    {
                        "type": "input_image",
                        "image_url": f"data:image/png;base64,{image_b64}",
                    },
                ],
            }
        ],
        questions=[
            {
                "type": "choice",
                "name": "category",
                "instructions": (
                    "Choose closed if the transparent lid is closed. "
                    "Choose opened if the transparent lid is raised away from the black box. "
                    "Choose unclear if it cannot be determined from the image."
                ),
                "choices": [{"value": label, "description": label} for label in labels],
            }
        ],
    )

    answer = decision.answers[0]
    if answer.type == "refusal":
        print(f"Refused: {answer.name}", file=sys.stderr)
        return None

    print(
        f"{datetime.now().time().isoformat(timespec='milliseconds')} "
        f"Predicted: {answer.choice} (confidence: {answer.confidence:.3f})"
    )
    return [(probability.value, probability.probability) for probability in answer.probabilities]


def display_image(image: np.ndarray, probabilities: list[tuple[str, float]] | None) -> None:
    """Show a frame with its probabilities."""
    if probabilities is not None:
        lines = ["Probabilities:"] + [
            f"  {label}: {probability:.3f}" for label, probability in probabilities
        ]
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_sizes = [cv2.getTextSize(line, font, PROBABILITY_FONT_SCALE, 1) for line in lines]
        line_height = max(height + baseline for (_, height), baseline in text_sizes) + PROBABILITY_LINE_GAP
        for index, (line, ((width, _), baseline)) in enumerate(zip(lines, text_sizes)):
            position = (
                image.shape[1] - PROBABILITY_MARGIN - width,
                image.shape[0] - PROBABILITY_MARGIN - baseline - (len(lines) - index - 1) * line_height,
            )
            cv2.putText(image, line, position, font, PROBABILITY_FONT_SCALE, (0, 255, 0), 1, cv2.LINE_AA)

    cv2.imshow("OpenAI Decision API Sample", image)


def main() -> None:
    """Capture and classify RealSense frames until 'q' is pressed."""
    client = OpenAI()
    pipeline = setup_realsense()

    try:
        while True:
            frame = get_frame(pipeline)
            if frame is None:
                print("Failed to capture.")
                continue

            probabilities = classify_image(frame, client)
            display_image(frame, probabilities)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()