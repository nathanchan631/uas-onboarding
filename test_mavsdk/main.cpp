// main.cpp
#include <iostream>
#include <string>
#include <chrono>
#include <cstdint>
#include <future>
#include <memory>
#include <thread>
#include <mavsdk/mavsdk.h>
#include <mavsdk/plugins/action/action.h>
#include <mavsdk/plugins/telemetry/telemetry.h>
#include <mavsdk/plugins/gimbal/gimbal.h>
#include <opencv2/opencv.hpp>

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
    const std::string file_path = "./capture.png";
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
    auto gimbal = Gimbal{system.value()};
    auto action = Action{system.value()};

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
    sleep_for(seconds(5));

    auto res = gimbal.take_control(0, Gimbal::ControlMode::Primary);
    if (res != Gimbal::Result::Success) {
        std::cerr << "Failed to take control: " << res << "\n";
        return 1;
    }

    // 2) set angles: roll=0°, pitch=-90° (point straight down), yaw=0°
    res = gimbal.set_angles(
        /*gimbal_id*/ 0,
        /*roll_deg*/  0.0f,
        /*pitch_deg*/ -90.0f,
        /*yaw_deg*/    0.0f,
        /*mode*/      Gimbal::GimbalMode::YawFollow,
        /*send_mode*/ Gimbal::SendMode::Once
    );  
    if (res != Gimbal::Result::Success) {
        std::cerr << "Failed to set angles: " << (res) << "\n";
        return 1;
    }

    // leave the gimbal down for 5 seconds...
    sleep_for(seconds(5));
    
    cv::Mat frame = take_photo();
    bool ok = cv::imwrite(file_path, frame);
    if (!ok) {
        std::cerr << "[saveDroneFrame] Failed to write " << file_path << "\n";
    }
    std::cout << "Saved Image to: " << file_path << "\n";
    //auto offset = get_offset(frame);
    offset = 1;
    if (offset) {
        std::cout << "Offset X: " << offset->x
                  << ", Offset Y: " << offset->y << "\n";
    } else {
        std::cout << "No valid red dot detected\n";
    }

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
