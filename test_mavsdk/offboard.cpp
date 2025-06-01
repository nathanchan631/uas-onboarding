#include <mavsdk/mavsdk.h>
#include <mavsdk/plugins/offboard/offboard.h>
#include <mavsdk/plugins/action/action.h>
#include <mavsdk/plugins/telemetry/telemetry.h>

#include <chrono>
#include <iostream>
#include <thread>

using namespace mavsdk;
using namespace std::chrono_literals;

int main() {
    Mavsdk mavsdk{Mavsdk::Configuration{ComponentType::GroundStation}};
    ConnectionResult connection_result = mavsdk.add_any_connection("udp://:14540");
    if (connection_result != ConnectionResult::Success) {
        std::cerr << "Connection failed: " << connection_result << std::endl;
        return 1;
    }

    std::shared_ptr<System> system;
    std::cout << "Waiting for drone..." << std::endl;
    mavsdk.subscribe_on_new_system([&mavsdk, &system]() {
        for (auto& sys : mavsdk.systems()) {
            if (sys->is_connected()) {
                system = sys;
                break;
            }
        }
    });

    // Wait until system is found
    while (!system) {
        std::this_thread::sleep_for(500ms);
    }

    auto telemetry = Telemetry{system};
    auto action = Action{system};
    auto offboard = Offboard{system};

    // Wait for system to be ready
    int cnt = 0;
    while (!telemetry.health_all_ok() && cnt < 5) {
        std::cout << "Waiting for system health..." << std::endl;
        std::this_thread::sleep_for(1s);
        cnt++;
    }

    // Arm and takeoff
    std::cout << "Arming..." << std::endl;
    if (action.arm() != Action::Result::Success) {
        std::cerr << "Failed to arm" << std::endl;
        return 1;
    }

    std::cout << "Taking off..." << std::endl;
    action.set_takeoff_altitude(5.0);
    if (action.takeoff() != Action::Result::Success) {
        std::cerr << "Failed to takeoff" << std::endl;
        return 1;
    }

    std::this_thread::sleep_for(5s);

    // Set initial velocity command to "start offboard mode"
    Offboard::VelocityNedYaw stay_still{};
    stay_still.north_m_s = 0.0f;
    stay_still.east_m_s = 0.0f;
    stay_still.down_m_s = 0.0f;
    stay_still.yaw_deg = 0.0f;
    offboard.set_velocity_ned(stay_still);

    std::cout << "Starting Offboard mode..." << std::endl;
    if (offboard.start() != Offboard::Result::Success) {
        std::cerr << "Failed to start Offboard mode" << std::endl;
        return 1;
    }

    // Fly forward at 3 m/s for 10 seconds
    Offboard::VelocityNedYaw forward{};
    forward.north_m_s = 3.0f;  // 3 m/s forward (North)
    forward.east_m_s = 0.0f;
    forward.down_m_s = 0.0f;
    forward.yaw_deg = 0.0f;

    std::cout << "Flying forward at 3 m/s for 10 seconds..." << std::endl;
    offboard.set_velocity_ned(forward);
    std::this_thread::sleep_for(10s);

    // Stop movement
    std::cout << "Stopping..." << std::endl;
    offboard.set_velocity_ned(stay_still);
    std::this_thread::sleep_for(1s);

    // Stop Offboard and land
    offboard.stop();
    std::cout << "Landing..." << std::endl;
    if (action.land() != Action::Result::Success) {
        std::cerr << "Failed to land" << std::endl;
        return 1;
    }

    std::this_thread::sleep_for(10s);
    std::cout << "Done." << std::endl;
    return 0;
}
