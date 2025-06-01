#include <mavsdk/mavsdk.h>
#include <mavsdk/plugins/offboard/offboard.h>
#include <mavsdk/plugins/action/action.h>
#include <mavsdk/plugins/telemetry/telemetry.h>
#include <mavsdk/geometry.h>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <thread>

using namespace mavsdk;
using namespace mavsdk::geometry;
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

double calculate_error(Telemetry::Position current_pos, Telemetry::Position target_pos, const CoordinateTransformation &transform) {
    auto local_coord = transform.local_from_global({current_pos.latitude_deg, current_pos.longitude_deg});
    auto target_coord = transform.local_from_global({target_pos.latitude_deg, target_pos.longitude_deg});

    double error_x = target_coord.east_m - local_coord.east_m;    
    double error_y = target_coord.north_m - local_coord.north_m;

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
    
    // set target position
    const Telemetry::Position target{0.0002029, 0.0003410};
    const float error_threshold = 0.5f;
    const CoordinateTransformation transform{{target.latitude_deg, target.longitude_deg}};

    PID pid_x(1, 0, 0);
    PID pid_y(1, 0, 0);
    Telemetry::Position current_pos = telemetry.position();

    // Telemetry::PositionVelocityNed current_pos = telemetry.position_velocity_ned();
    while (calculate_error(current_pos, target, transform) > error_threshold) {
        current_pos = telemetry.position();
        std::cout << "Current position: " << current_pos.longitude_deg << ", " << current_pos.latitude_deg << std::endl;

        // PID controller
        double dt = 0.02; // time step
        double error = calculate_error(current_pos, target, transform);

        auto transformed_target = transform.local_from_global({target.latitude_deg, target.longitude_deg});
        auto transformed_current = transform.local_from_global({current_pos.latitude_deg, current_pos.longitude_deg});

        std::cout << "Current position transformed: " << transformed_current.east_m << ", " << transformed_current.north_m << std::endl;

        double control_signal_x = pid_x.compute(transformed_target.east_m, transformed_current.east_m, dt);
        double control_signal_y = pid_y.compute(transformed_target.north_m, transformed_current.north_m, dt);
        std::cout << "Control signal x: " << control_signal_x << ", Control signal y: " << control_signal_y << std::endl;
        std::cout << "Error: " << error << std::endl;

        control_signal_x = std::clamp(control_signal_x, -5.0, 5.0);
        control_signal_y = std::clamp(control_signal_y, -5.0, 5.0);

        // Set velocity command
        Offboard::VelocityNedYaw velocity_command{};
        velocity_command.north_m_s = control_signal_y;
        velocity_command.east_m_s = control_signal_x;
        velocity_command.down_m_s = 0.0f;
        velocity_command.yaw_deg = 0.0f;
        offboard.set_velocity_ned(velocity_command);
        if (offboard.start() != Offboard::Result::Success) {
            std::cerr << "Failed to start Offboard mode" << std::endl;
            return 1;
        }
        // Sleep for a short duration to allow the drone to respond
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }

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
}