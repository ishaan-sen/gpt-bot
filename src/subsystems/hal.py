import pigpio
from subsystems.drive import Drive

class HAL:
    def __init__(self):
        self.pi = pigpio.pi()
        self.drive = Drive(self.pi)
