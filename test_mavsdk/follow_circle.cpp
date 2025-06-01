
#include <mavsdk/mavsdk.h>
#include <mavsdk/plugins/offboard/offboard.h>
#include <mavsdk/plugins/action/action.h>
#include <mavsdk/plugins/telemetry/telemetry.h>
#include <mavsdk/plugins/gimbal/gimbal.h>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <thread>

#include "camera.h"
#include "detector.h"

using namespace mavsdk;
using namespace std::chrono_literals;

struct PID {
    double kp;
    double ki;
    double kd;
    double prev_error;
    double integral;

    PID(double p, double i, double d) : kp(p), ki(i), kd(d), prev_error(0), integral(0) {}

    double compute(double setpoint, double measured_value, double dt) {
        double error = setpoint - measured_value;
        integral += error * dt;
        double derivative = (error - prev_error) / dt;
        prev_error = error;

        return kp * error + ki * integral + kd * derivative;
    }
};

double calculate_error(Telemetry::PositionVelocityNed current_pos, float target_x, float target_y) {
    double error_x = target_x - current_pos.position.east_m;
    double error_y = target_y - current_pos.position.north_m;
    return std::sqrt(error_x * error_x + error_y * error_y);
}

double calculate_error(double current_x, double current_y, double target_x, double target_y) {
    double error_x = target_x - current_x;
    double error_y = target_y - current_y;
    return std::sqrt(error_x * error_x + error_y * error_y);
}

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
    auto gimbal = Gimbal{system};

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
    // take off
    std::cout << "Taking off..." << std::endl;
    action.set_takeoff_altitude(5.0);
    if (action.takeoff() != Action::Result::Success) {
        std::cerr << "Failed to takeoff" << std::endl;
        return 1;
    }
    std::this_thread::sleep_for(5s);
    std::cout << "Finished takeoff" << std::endl;

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

    std::cout << "Offboard mode started, starting PID" << std::endl;


    double current_x = 0.0;
    double current_y = 0.0;
    double current_z = 0.0;
    const float target_x = 0.0f;
    const float target_y = 0.0f;
    const float target_z = 5.0f;
    const float error_threshold = 10.f;
    PID pid_x(0.01, 0, 0);
    PID pid_y(0.01, 0, 0);
    PID pid_z(1, 0, 0);
    double prev_time = 0;

    Offboard::VelocityBodyYawspeed velocity_command{};
    velocity_command.forward_m_s = 0;
    velocity_command.right_m_s = 0;
    velocity_command.down_m_s = 0.0f;
    velocity_command.yawspeed_deg_s = 0.0f;
    offboard.set_velocity_body(velocity_command);
    do {
        // test photo
        std::cout << "Taking photo..." << std::endl;
        cv::Mat photo = take_photo();
        if (photo.empty()) {
            std::cerr << "Failed to take photo" << std::endl;
            return 1;
        }
        std::cout << "Photo taken" << std::endl;
        std::optional<cv::Point2d> offset = get_offset(photo);
        if (!offset) {
            std::cerr << "Failed to detect red dot" << std::endl;
            return 1;
        }
        std::cout << "Red dot detected at: " << offset->x << ", " << offset->y << std::endl;

        current_x = offset->x; 
        current_y = offset->y;
        current_z = telemetry.altitude().altitude_relative_m;
        std::cout << "Current position: " << current_x << ", " << current_y << ", " << current_z << std::endl;

        double current_time_sec = std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
        double dt = current_time_sec - prev_time;
        prev_time = current_time_sec;
        double error = calculate_error(current_x, current_y, target_x, target_y);
        double control_signal_x = -pid_x.compute(target_x, current_x, dt);
        double control_signal_y = -pid_y.compute(target_y, current_y, dt);
        double control_signal_z = -pid_z.compute(target_z, current_z, dt);
        std::cout << "Control signal x: " << control_signal_x << ", Control signal y: " << control_signal_y << ", Control signal z: " << control_signal_z << std::endl;
        std::cout << "Error: " << error << std::endl;

        // not working:
        // currently going the wrong way
        // perhaps need to rotate body frame/frame of camera?
        // but camera seems to be in the body frame so i'm confused

        control_signal_x = std::clamp(control_signal_x, -5.0, 5.0);
        control_signal_y = std::clamp(control_signal_y, -5.0, 5.0);
        control_signal_z = std::clamp(control_signal_z, -5.0, 5.0);

        // // Set velocity command
        Offboard::VelocityBodyYawspeed velocity_command{};
        velocity_command.forward_m_s = control_signal_y;
        velocity_command.right_m_s = control_signal_x;
        velocity_command.down_m_s = control_signal_z;
        velocity_command.yawspeed_deg_s = 0.0f;
        offboard.set_velocity_body(velocity_command);
        std::this_thread::sleep_for(20ms);
    } while (calculate_error(current_x, current_y, target_x, target_y) > error_threshold);

    // Stop movement
    std::cout << "Stopping..." << std::endl;
    offboard.set_velocity_ned(stay_still);
    std::this_thread::sleep_for(1s);
    offboard.stop();

    std::cout << "Landing..." << std::endl;
    if (action.land() != Action::Result::Success) {
        std::cerr << "Failed to land" << std::endl;
        return 1;
    }

    std::this_thread::sleep_for(10s);
    std::cout << "Done." << std::endl;
}