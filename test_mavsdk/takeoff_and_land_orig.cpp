//
// Simple example to demonstrate how takeoff and land using MAVSDK.
//

#include <chrono>
#include <cstdint>
#include <mavsdk/mavsdk.h>
#include <mavsdk/plugins/action/action.h>
#include <mavsdk/plugins/telemetry/telemetry.h>
#include <iostream>
#include <future>
#include <memory>
#include <thread>

using namespace mavsdk;
using std::chrono::seconds;
using std::this_thread::sleep_for;

void usage(const std::string& bin_name)
{
    std::cerr << "Usage : " << bin_name << " <connection_url>\n"
              << "Connection URL format should be :\n"
              << " For TCP server: tcpin://<our_ip>:<port>\n"
              << " For TCP client: tcpout://<remote_ip>:<port>\n"
              << " For UDP server: udp://<our_ip>:<port>\n"
              << " For UDP client: udp://<remote_ip>:<port>\n"
              << " For Serial : serial://</path/to/serial/dev>:<baudrate>]\n"
              << "For example, to connect to the simulator use URL: udpin://0.0.0.0:14540\n";
}

int main(int argc, char** argv)
{
    if (argc != 2) {
        usage(argv[0]);
        return 1;
    }

    Mavsdk mavsdk{Mavsdk::Configuration{ComponentType::GroundStation}};
    ConnectionResult connection_result = mavsdk.add_any_connection(argv[1]);

    if (connection_result != ConnectionResult::Success) {
        std::cerr << "Connection failed: " << connection_result << '\n';
        return 1;
    }

    auto system = mavsdk.first_autopilot(3.0);
    if (!system) {
        std::cerr << "Timed out waiting for system\n";
        return 1;
    }

    // Instantiate plugins.
    auto telemetry = Telemetry{system.value()};
    auto action = Action{system.value()};

    // We want to listen to the altitude of the drone at 1 Hz.
    const auto set_rate_result = telemetry.set_rate_position(1.0);
    if (set_rate_result != Telemetry::Result::Success) {
        std::cerr << "Setting rate failed: " << set_rate_result << '\n';
        return 1;
    }

    // Set up callback to monitor altitude while the vehicle is in flight
    telemetry.subscribe_position([](Telemetry::Position position) {
        std::cout << "Altitude: " << position.relative_altitude_m << " m\n";
    });

    // Telemetry::Health health = telemetry.health();

    // if (!health.is_gyrometer_calibration_ok) {
    //     std::cout << "Gyrometer calibration not OK" << std::endl;
    // }
    // if (!health.is_accelerometer_calibration_ok) {
    //     std::cout << "Accelerometer calibration not OK" << std::endl;
    // }
    // if (!health.is_magnetometer_calibration_ok) {
    //     std::cout << "Magnetometer calibration not OK" << std::endl;
    // }
    // if (!health.is_local_position_ok) {
    //     std::cout << "Local position estimate not OK" << std::endl;
    // }
    // if (!health.is_global_position_ok) {
    //     std::cout << "Global position estimate not OK" << std::endl;
    // }
    // if (!health.is_home_position_ok) {
    //     std::cout << "Home position not set" << std::endl;
    // }


    // Check until vehicle is ready to arm
    int cnt = 0;
    while (telemetry.health_all_ok() != true && cnt < 5) {
        Telemetry::Health health = telemetry.health();

        std::cout << "--- Health Report ---" << std::endl;
        std::cout << "Gyro calib: " << health.is_gyrometer_calibration_ok << std::endl;
        std::cout << "Accel calib: " << health.is_accelerometer_calibration_ok << std::endl;
        std::cout << "Magnet calib: " << health.is_magnetometer_calibration_ok << std::endl;
        std::cout << "Local position: " << health.is_local_position_ok << std::endl;
        std::cout << "GPS fix (global): " << health.is_global_position_ok << std::endl;
        std::cout << "Home position: " << health.is_home_position_ok << std::endl;
        std::cout << "----------------------" << std::endl;
    
        std::this_thread::sleep_for(std::chrono::seconds(1));
        cnt++;
    }

    // Arm vehicle
    std::cout << "Arming...\n";
    const Action::Result arm_result = action.arm();

    if (arm_result != Action::Result::Success) {
        std::cerr << "Arming failed: " << arm_result << '\n';
        return 1;
    }

    // Take off
    std::cout << "Taking off...\n";
    const Action::Result takeoff_result = action.takeoff();
    if (takeoff_result != Action::Result::Success) {
        std::cerr << "Takeoff failed: " << takeoff_result << '\n';
        return 1;
    }

    // Let it hover for a bit before landing again.
    sleep_for(seconds(10));

    std::cout << "Landing...\n";
    const Action::Result land_result = action.land();
    if (land_result != Action::Result::Success) {
        std::cerr << "Land failed: " << land_result << '\n';
        return 1;
    }

    // Check if vehicle is still in air
    while (telemetry.in_air()) {
        std::cout << "Vehicle is landing...\n";
        sleep_for(seconds(1));
    }
    std::cout << "Landed!\n";

    // We are relying on auto-disarming but let's keep watching the telemetry for a bit longer.
    sleep_for(seconds(3));
    std::cout << "Finished...\n";

    return 0;
}