import pigpio

class Drive:
    def __init__(self, pi: pigpio.pi, drive_pin=18, servo_pin=19, drive_freq=1000, servo_min=1000, servo_max=2000):
        self.pi = pi
        if not self.pi.connected:
            raise IOError("Could not connect to pigpio daemon")
        self.drive_pin = drive_pin
        self.servo_pin = servo_pin
        self.drive_freq = drive_freq
        self.servo_min = servo_min
        self.servo_max = servo_max
        self.pi.set_mode(self.servo_pin, pigpio.OUTPUT)
    
    def set_drive_speed(self, speed):
        self.pi.hardware_PWM(self.drive_pin, self.drive_freq, int(speed * 1000000))
    
    def set_steering_angle(self, angle):
        # angle: 0 to 180 degrees
        pulse_width = int(((angle / 180.0) * (self.servo_max - self.servo_min)) + self.servo_min)
        self.pi.set_servo_pulsewidth(self.servo_pin, pulse_width)
    
    def drive(self, speed, direction):
        self.set_drive_speed(speed)
        angle = ((direction + 1) / 2.0) * 180
        self.set_steering_angle(angle)
    
    def stop(self):
        self.pi.hardware_PWM(self.drive_pin, 0, 0)
    
    def cleanup(self):
        self.stop()
        self.pi.stop()

