import asyncio
import os
from mavsdk import System
from mavsdk.offboard import (OffboardError, PositionNedYaw, VelocityNedYaw)
from mavsdk.gimbal import (GimbalMode, ControlMode)
from camera import Video
import cv2
import requests
from detector import detect

async def run():
    drone = System()
    camera = Video()
    offset = {}
    position = {"x": 0.0, "y": 0.0}
    await drone.connect(system_address="udp://:14540")
    print("Waiting for drone to connect...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print(f"-- Connected to drone!")
            break



    altitude = 1.5
    print(f"Setting takeoff altitude to {altitude} m")

    print("Arming drone...")
    await drone.action.arm()

    print("Taking off...")
    await drone.offboard.set_position_ned(
        PositionNedYaw(0.0, 0.0, 0.0, 0.0))
    await drone.offboard.start()
    await drone.offboard.set_position_velocity_ned(
        PositionNedYaw(0.0, 0.0, -1 * altitude, 0.0),
        VelocityNedYaw(0.0, 0.0, -1.0, 0.0)
    )
    await asyncio.sleep(5)

    print("Pointing camera gimbal straight down")
    await drone.gimbal.take_control(ControlMode.PRIMARY)
    await drone.gimbal.set_mode(GimbalMode.YAW_LOCK)
    await drone.gimbal.set_pitch_and_yaw(-90,0)
    await asyncio.sleep(5)

    do_continue = True
    while(do_continue):
        do_continue = await take_photo_and_move(drone, camera, -1 * altitude, position)

    await drone.offboard.stop()
    await drone.action.land()

async def take_photo_and_move(drone, camera, altitude, position):
    try:
        print("Taking photo...")
        offset = {}
        frame = camera.frame()
        image_location = '../images/image.jpg'
        cv2.imwrite(image_location, frame)
        print("Sending photo to vision...")
        res = detect(image_location)
        offset['x'] = res[0]
        offset['y'] = res[1]
        print(f"{offset['x']} {offset['y']}")
        
        """ TODO """

    except Exception as error:
        print(f"{error}")
        print("Didn't work")
        return None

if __name__ == "__main__":
    asyncio.run(run())

