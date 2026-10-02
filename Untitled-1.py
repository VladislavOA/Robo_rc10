import time
from rc10_api.rc_api import RobotApi



robot = RobotApi(ip="10.10.10.10", read_only=False, timeout=5)


time.sleep(0.3)

robot.motion.scale_setup.set(velocity=0.02, acceleration=0.02)
robot.controller_state.set("run", await_sec=10)

for _ in range(20):  # примерно 0.2 сек движения
    robot.motion.linear.jog_once("Z", "+")
    time.sleep(0.01)

robot.motion.linear.jog_once_all_axis(["0", "0", "0", "0", "0", "0"])

robot._shutdown_api_sockets()

print("Готово")