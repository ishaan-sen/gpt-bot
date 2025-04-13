import base64
import cv2
import json
from openai import OpenAI
import matplotlib.pyplot as plt
import time


query = '''
You are controlling a rover exploring a house to find a basketball. Two images are provided:
The first image is the current camera view, and the second image is the view from the previous step.
Use the current image to decide one movement command from the options: "left", "right", or "forward"—choose the best direction to move the rover toward the basketball while avoiding obstacles.
You may use the previous image to compare scenes, avoid revisiting the same area, or detect changes.
Also provide a brief, one-sentence description of the current scene.
Respond with a valid JSON object containing exactly these keys: "scene_description" and "command".
'''

class GPT:
    def __init__(self):
        self.client = OpenAI()

        self.cap = cv2.VideoCapture(0)
        time.sleep(0.5)


    def capture_image(self):
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to capture image from camera.")
        ret, buffer = cv2.imencode(".jpg", frame)
        if not ret:
            raise RuntimeError("Failed to encode image.")

        # plt.imshow(frame)
        # plt.show()
        
        return base64.b64encode(buffer).decode("utf-8")

    def generate_command(self, image, last_image):
        response = self.client.responses.create( 
            model="gpt-4o",
            input=[
                { # type: ignore
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": query},
                        {
                            "type": "input_image",
                            "image_url": f"data:image/jpeg;base64,{image}",
                            "detail": "low",
                        },
                        {
                            "type": "input_image",
                            "image_url": f"data:image/jpeg;base64,{last_image}",
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

        if "command" not in data or "scene_description" not in data:
            raise ValueError(
                "JSON must contain both 'command' and 'scene_description' keys"
            )

        command = data["command"]

        if not isinstance(command, str):
            raise ValueError("'command' must be a string")

        return command

    def cleanup(self):
        self.cap.release()
        
