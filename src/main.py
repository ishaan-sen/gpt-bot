import time
import atexit

from subsystems.gpt import GPT
from subsystems.hal import HAL


class Robot:
    def __init__(self):
        self.gpt = GPT()
        # self.hal = HAL()

        self.last_image = self.gpt.capture_image()



    def periodic(self):
        image = self.gpt.capture_image()
        output = self.gpt.generate_command(image, self.last_image)
        self.last_image = image
        print(output)
        command = self.gpt.parse_command(output)
        print(command)

        if command == "forward":
            drive_input = (0.6, 0.0)
        elif command == "left":
            drive_input = (0.4, -1.0)
        elif command == "right":
            drive_input = (0.4, 1.0)
        elif command == "stop":
            drive_input = (0.0, 0.0)
        else:
            drive_input = (0.0, 0.0)
            raise ValueError(command)

        # self.hal.drive.drive(*drive_input)
        print(command, drive_input)

    def cleanup(self):
        pass
        # self.hal.drive.cleanup()


if __name__ == "__main__":
    robot = Robot()

    atexit.register(robot.cleanup)

    loop_time = 3
    last_time = time.time()

    while True:
        now = time.time()
        if now - last_time > loop_time:
            last_time = now
            robot.periodic()
            time.sleep(0.1)
