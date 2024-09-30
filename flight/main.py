import asyncio
import os
from mavsdk import System
from mavsdk.offboard import (OffboardError, VelocityBodyYawspeed)
from mavsdk.gimbal import (GimbalMode, ControlMode)
from camera import Video
import cv2
import requests

async def run():
    drone = System()
    camera = Video()
    offset = {}
    await drone.connect(system_address="udp://:14540")
    print("Waiting for drone to connect...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print(f"-- Connected to drone!")
            break



    takeoff_alt = 1.5
    print(f"Setting takeoff altitude to {takeoff_alt} m")
    await drone.action.set_takeoff_altitude(takeoff_alt)

    print("Arming drone...")
    await drone.action.arm()

    print("Taking off...")
    await drone.action.takeoff()
    await check_altitude(drone, takeoff_alt)
    print("Pointing camera gimbal straight down")
    await drone.gimbal.take_control(ControlMode.PRIMARY)
    await drone.gimbal.set_mode(GimbalMode.YAW_LOCK)
    await drone.gimbal.set_pitch_and_yaw(-90,0)
    await asyncio.sleep(5)

    frame = camera.frame()
    cv2.imwrite('../images/before.jpg', frame)
    try:
        do_continue = await take_photo_and_move(drone, camera)
        while(do_continue):
            do_continue = await take_photo_and_move(drone, camera)

    except Exception:
        print("for loop died")

    frame = camera.frame()
    cv2.imwrite('../images/after.jpg', frame)
    await drone.action.land()


async def check_altitude(drone, alt):
    async for pos in drone.telemetry.position():
        if (pos.relative_altitude_m >= (alt - 0.3)):
            break
    return

async def take_photo_and_move(drone, camera):
    try:
        print("Taking photo...")
        offset = {}
        frame = camera.frame()
        image_location = '../images/image.jpg'
        cv2.imwrite(image_location, frame)
        print("Sending photo to vision...")
        res = requests.post('http://localhost:8003/odlc', json={'img_name': "image.jpg"})
        offset['x'] = res.json()[0]
        offset['y'] = res.json()[1]
        print(f"{offset['x']} {offset['y']}")
        await drone.offboard.set_velocity_body(
        VelocityBodyYawspeed(-1 * (offset['x'] / 1000),
                             -1 * (offset['y'] / 1000),
                             0,
                             0))
        await drone.offboard.start()
        await asyncio.sleep(1.5)
        await drone.offboard.stop()
        await asyncio.sleep(2)
        return (abs(offset['x']) >= 10 or abs(offset['y']) >= 10)

    except Exception as error:
        print(f"{error}")
        print("Didn't work")
        return None

if __name__ == "__main__":
    asyncio.run(run())

