#include <mavsdk/mavsdk.h>
#include <mavsdk/plugins/offboard/offboard.h>
#include <mavsdk/plugins/action/action.h>
#include <mavsdk/plugins/telemetry/telemetry.h>
#include <mavsdk/plugins/mission/mission.h>
#include <mavsdk/plugins/gimbal/gimbal.h>
#include <curl/curl.h>
#include <json/json.h>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <thread>

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

bool upload_mission(Mission &mission) {
    // push back mission item to go to target point
    std::vector<Mission::MissionItem> mission_items;
    Mission::MissionItem new_item;
    new_item.latitude_deg = 0.0002029;
    new_item.longitude_deg = 0.0003410;
    new_item.relative_altitude_m = 5.0f;
    new_item.speed_m_s = 2.0f;
    new_item.is_fly_through = false;              // <-- this makes it STOP
    new_item.acceptance_radius_m = 1.0f;          // <-- how close it needs to be
    new_item.camera_action = Mission::MissionItem::CameraAction::None;
    mission_items.push_back(new_item);
    Mission::MissionPlan missionPlan{};
    missionPlan.mission_items = mission_items;

    std::cout << "Uploading mission..." << std::endl;

    // clear any previous missions
    const Mission::Result clear_result = mission.clear_mission();
    if (clear_result != Mission::Result::Success) {
        std::cerr << "Mission clear failed: " << clear_result << std::endl;
        return false;
    }

    // upload red circle waypoint
    const Mission::Result upload_result = mission.upload_mission(missionPlan);
    if (upload_result != Mission::Result::Success) {
        std::cerr << "Mission upload failed: " << upload_result << ", exiting.\n";
        return false;
    }

    return true;
}

bool bring_down_gimbal(Gimbal &gimbal) {
    auto res = gimbal.take_control(0, Gimbal::ControlMode::Primary);
    if (res != Gimbal::Result::Success) {
        std::cerr << "Failed to take control: " << res << "\n";
        return false;
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
        return false;
    }

    return true;
}

struct Point2d {
	double x, y;
};

// Callback function to write received data into a std::string
// This function is called by libcurl as data arrives from the server.
size_t write_callback(void *contents, size_t size, size_t nmemb, void *userp) {
    // Calculate the real size of the data received in this chunk
    size_t realsize = size * nmemb;
    // Cast the userp pointer back to a std::string* and append the contents
    ((std::string*)userp)->append((char*)contents, realsize);
    // Return the number of bytes processed; libcurl expects this
    return realsize;
}

static Point2d post_request(std::string shared_mem_name) {
  //Make curl object
  CURL *curl;
  CURLcode request;

  // Intialize curl
  curl_global_init(CURL_GLOBAL_DEFAULT);
  curl = curl_easy_init();

  // String to store the HTTP response body
  std::string response_buffer;

  Point2d ret;
  ret.x = 0;
  ret.y = 0;

	if (curl) {
	    // Set url to make post to
	    curl_easy_setopt(curl, CURLOPT_URL, "http://localhost:8003/odlc");

	    // Set request to be a POST
	    curl_easy_setopt(curl, CURLOPT_POST, 1L);

	    // Set JSON data to send the name of the shared memory object and the
	    // telemetry data taken at the same time
	    std::string json_string = "";
	    json_string += "{\"img_name\": \"" + shared_mem_name + "}";
	    const char *json_data = json_string.c_str();
	    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, json_data);

	    // Set content type header to json
	    struct curl_slist *headers = nullptr;
	    headers = curl_slist_append(headers, "Content-Type: application/json");
	    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);

	    // Set the callback function to write the response data
	    // libcurl will call 'write_callback' whenever it receives data
	    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_callback);
	    // Pass the 'response_buffer' (address of our string) to the callback
	    // This allows the callback to write into our string
	    curl_easy_setopt(curl, CURLOPT_WRITEDATA, (void *)&response_buffer);

	    // Send the request
	    request = curl_easy_perform(curl);

	    // Output if the request was successful or failed
	    if (request != CURLE_OK) {
	      std::cerr << "Request failed: " << curl_easy_strerror(request)
			<< std::endl;
	    } else {
	      std::cout << "Request success." << std::endl;
	      // Print the captured response
	      std::cout << "Response:\n" << response_buffer << std::endl;
	    }

	    // Clean up
	    curl_easy_cleanup(curl);
	    curl_slist_free_all(headers);

	    // parse response
	    Json::Value root;
		Json::Reader reader;
		reader.parse(response_buffer, root);
		ret.x = root[0].asDouble();
		ret.y = root[1].asDouble();
	  }

  curl_global_cleanup();

  return ret;
}

bool pid_to_red_circle(Offboard &offboard, Telemetry &telemetry) {
    double current_x = 0.0;
    double current_y = 0.0;
    double current_z = 0.0;
    const float target_x = 0.0f;
    const float target_y = 0.0f;
    const float target_z = 5.0f;
    const float error_threshold = 10.f;
    PID pid_x(0.008, 0, 0.003);
    PID pid_y(0.008, 0, 0.003);
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
        // cv::Mat photo = take_photo();
	
	// take_photo(); // takes photo
	/*
        if (photo.empty()) {
            std::cerr << "Failed to take photo" << std::endl;
            return false;
        }
	*/
        std::cout << "Photo taken" << std::endl;
	/*
	 Post request here, set offset equal to the json output of the post request to the network
	*/

	std::optional<Point2d> offset = post_request("img_name");
        if (!offset) {
            std::cerr << "Failed to detect red dot" << std::endl;
            return false;
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

    return true;
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
    auto mission = Mission{system};

    // Wait for system to be ready
    int cnt = 0;
    while (!telemetry.health_all_ok() && cnt < 5) {
        std::cout << "Waiting for system health..." << std::endl;
        std::this_thread::sleep_for(1s);
        cnt++;
    }

    // upload all missions
    if (!upload_mission(mission)) {
        return 1;
    }

    // sleep for 5s to wait for all missions to be uploaded
    std::this_thread::sleep_for(5s);

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
    std::cout << "sleeping..." << std::endl;
    std::this_thread::sleep_for(10s);
    std::cout << "Finished takeoff" << std::endl;

    // bring down gimbal
    if (!bring_down_gimbal(gimbal)) {
        return 1;
    }

    // leave the gimbal down for 5 seconds...
    std::this_thread::sleep_for(5s);

    // run mission
    // std::cout << "Starting mission..." << std::endl;
    // Mission::Result start_mission_result = mission.start_mission();
    // if (start_mission_result != Mission::Result::Success) {
    //     std::cerr << "Starting mission failed: " << start_mission_result << '\n';
    //     return 1;
    // }
    // while (!mission.is_mission_finished().second) {
    //     // std::cout << "Mission finished: " << (mission.is_mission_finished().first == Mission::Result::Success) << " " << mission.is_mission_finished().second << std::endl;
    //     std::this_thread::sleep_for(1s);
    // }
    // std::cout << "Mission is finished" << std::endl;

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

    pid_to_red_circle(offboard, telemetry);

    // Stop movement
    std::cout << "Stopping..." << std::endl;
    offboard.set_velocity_ned(stay_still);
    std::this_thread::sleep_for(1s);
    // offboard.stop();

    std::cout << "Landing..." << std::endl;
    if (action.land() != Action::Result::Success) {
        std::cerr << "Failed to land" << std::endl;
        return 1;
    }

    std::this_thread::sleep_for(10s);
    std::cout << "Done." << std::endl;
}
