from pymycobot.mycobot import MyCobot
import time

PORT = "/dev/ttyUSB0"
BAUD = 1000000

mc = MyCobot(PORT, BAUD)

print("Connected.")
time.sleep(1)

print("Current angles:")
print(mc.get_angles())

print("Current coords:")
print(mc.get_coords())

print("Opening gripper...")
mc.set_gripper_state(0, 30)
time.sleep(2)

print("Moving down slightly...")
mc.send_coords([55.0, -64.8, 388.2, -94.1, -45.07, -87.26], 10, 0)
time.sleep(4)

print("Moving back up...")
mc.send_coords([55.0, -64.8, 408.2, -94.1, -45.07, -87.26], 10, 0)
time.sleep(4)

print("Closing gripper...")
mc.set_gripper_state(1, 30)
time.sleep(2)

print("Opening gripper...")
mc.set_gripper_state(0, 30)
time.sleep(2)

print("Final coords:")
print(mc.get_coords())

print("Done.")
