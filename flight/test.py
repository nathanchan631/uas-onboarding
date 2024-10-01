import asyncio
from mavsdk import System
from mavsdk.offboard import (PositionNedYaw, VelocityNedYaw)

async def run():
    drone = System()
    await drone.connect(system_address="udp://:14540")
    print("Waiting for drone to connect...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print(f"-- Connected to drone!")
            break
    await drone.action.arm()
    await drone.offboard.set_position_ned(PositionNedYaw(0.0, 0.0, 0.0, 0.0))
    await drone.offboard.start()
    await drone.offboard.set_position_velocity_ned(
        PositionNedYaw(0.0, 0.0, -3.0, 0.0),
        VelocityNedYaw(0.0, 0.0, -1.0, 0.0))
    await asyncio.sleep(10)

    await drone.offboard.set_position_velocity_ned(
        PositionNedYaw(50.0, 20.0, -3.0, 0.0),
        VelocityNedYaw(1.0, 1.0, 0.0, 0.0))
    await asyncio.sleep(20)
    await drone.offboard.stop()
    await drone.action.land()




if __name__ == "__main__":
    asyncio.run(run())
