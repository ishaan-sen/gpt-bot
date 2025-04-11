import base64
import cv2
import json
from openai import OpenAI


class GPT:
    def __init__(self):
        self.client = OpenAI()

    def capture_image(self, camera_index=0):
        cap = cv2.VideoCapture(camera_index)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise RuntimeError("Failed to capture image from camera.")
        ret, buffer = cv2.imencode(".jpg", frame)
        if not ret:
            raise RuntimeError("Failed to encode image.")
        return base64.b64encode(buffer).decode("utf-8")

    def generate_command(self, image):
        query_text = (
            "You are controlling a rover exploring inside a house, with the objective of finding friendly people to help. "
            "Explore and move around the house looking for people to serve. Below is a camera image from the rover. "
            "Your task is to analyze the image and output a single command for the robot from the options: "
            "'left', 'right', 'forward', or 'stop'. Additionally, if a human is present in the frame, describe "
            "their approximate location using natural language (e.g., 'left', 'center', 'bottom right'); "
            "if no human is detected, set the human_position value to null. Respond with a valid JSON object "
            "containing exactly two keys: 'command' and 'human_position'. For example: {\"command\": \"forward\", "
            '"human_position": "top left"} or {"command": "stop", "human_position": null}. Do not include '
            "any extra text."
        )

        response = self.client.responses.create(
            model="gpt-4o-mini",
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": query_text},
                        {
                            "type": "input_image",
                            "image_url": f"data:image/jpeg;base64,{image}",
                            "detail": "low",
                        },
                    ],
                }
            ],
        )
        return response.output_text

    def parse_command(self, response_text):
        if response_text.strip().startswith("```"):
            lines = response_text.strip().splitlines()
            lines = [line for line in lines if not line.strip().startswith("```")]
            response_text = "\n".join(lines)

        try:
            data = json.loads(response_text)
        except json.JSONDecodeError as e:
            raise ValueError("Response is not valid JSON") from e

        if "command" not in data or "human_position" not in data:
            raise ValueError(
                "JSON must contain both 'command' and 'human_position' keys"
            )

        command = data["command"]
        human_position = data["human_position"]

        if not isinstance(command, str):
            raise ValueError("'command' must be a string")
        if human_position is not None and not isinstance(human_position, str):
            raise ValueError("'human_position' must be a string or null")

        return command, human_position
