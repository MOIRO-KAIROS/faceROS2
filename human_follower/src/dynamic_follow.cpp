#include <rclcpp/rclcpp.hpp>
#include <vector>
#include <math.h>
#include <moiro_interfaces/srv/target_pose.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <moiro_interfaces/srv/target_depth.hpp>

#define R_VEL   0.05         // rotate velocity
#define F_VEL   0.05         // forward velocity

struct HumanPose {
    std::tuple<double, double, double> goal; // (x, y, z) 좌표
    bool valid = false;
};

bool init_flag = true;

geometry_msgs::msg::Twist velOutput;
HumanPose p;

class HumanFollower : public rclcpp::Node {
public:
    HumanFollower() : Node("human_follower"), MAX_DEPTH(1.5), MIN_DEPTH(1.0)  {
        RCLCPP_INFO(this->get_logger(), "Initialized node");

        MIN_Y = -0.05;
        MAX_Y = 0.05;
        
        pub_ = this->create_publisher<geometry_msgs::msg::Twist>("cmd_vel", 10);
        srv_client = this->create_client<moiro_interfaces::srv::TargetPose>("vision/target_pose");
        
        this->declare_parameter("max_depth", 1.5);
        MAX_DEPTH = this->get_parameter("max_depth").as_double();
        this->declare_parameter("min_depth", 1.0);
        MIN_DEPTH = this->get_parameter("min_depth").as_double();

        srv_depth = this->create_service<moiro_interfaces::srv::TargetDepth>(
            "target_depth", std::bind(&HumanFollower::depth_change, this, std::placeholders::_1, std::placeholders::_2) 
        );

        // 주기적으로 check_and_request 호출
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(500),
            std::bind(&HumanFollower::check_and_request, this)
        );
    }

private:
    void depth_change(const std::shared_ptr<moiro_interfaces::srv::TargetDepth::Request> request,
                      std::shared_ptr<moiro_interfaces::srv::TargetDepth::Response> response) {
        MAX_DEPTH = request-> max_depth;
        MIN_DEPTH = request-> min_depth;
        response->message = "Tareget Depth Max [" + std::to_string(MAX_DEPTH)+ "],  min [" + std::to_string(MIN_DEPTH) + ']';
    }
    void check_and_request() {
        if (!p.valid) {
            request_target();
        }
    }

    void request_target() {
        auto request = std::make_shared<moiro_interfaces::srv::TargetPose::Request>();
        request->prepared = true;
        auto future = srv_client->async_send_request(
            request, std::bind(&HumanFollower::handle_response, this, std::placeholders::_1)
        );
    }

    void handle_response(rclcpp::Client<moiro_interfaces::srv::TargetPose>::SharedFuture future) {
        try {
            auto response = future.get();
            p.goal = std::make_tuple(response->x, response->y, response->z);
            p.valid = true;

            if (response->status) {
                RCLCPP_INFO(this->get_logger(), "\033[93mI DETECT HUMAN\033[0m");
                RCLCPP_INFO(this->get_logger(), "Target Pose: x=%f, y=%f, z=%f", std::get<0>(p.goal), std::get<1>(p.goal), std::get<2>(p.goal));
                GettingHuman();
                if (init_flag){
                    init_flag = false;
                }
            } else {
                if (!init_flag)
                    RCLCPP_ERROR(this->get_logger(), "I LOST HUMAN");
                LostHuman();
            }

            p.valid = false;
        } catch (const std::exception &e) {
            p.valid = false;
            RCLCPP_ERROR(this->get_logger(), "Failed to get target pose: %s", e.what());
            LostHuman();
        }
    }

    void GettingHuman() {
        double person_x = std::get<0>(p.goal);
        double person_y = std::get<1>(p.goal);

        // 로봇의 회전 제어: y 값을 사용하여 카메라 중앙에 정렬
        if (person_y < MIN_Y) // -0.05
            velOutput.angular.z = - R_VEL;
        else if (person_y > MAX_Y) // 0.05
            velOutput.angular.z = R_VEL;
        else
            velOutput.angular.z = 0;
        
        // 로봇의 전진/후진 제어: x 값을 사용하여 특정 거리 유지
        if (person_x > MAX_DEPTH) {
            RCLCPP_INFO(this->get_logger(), "FORWARD");
            velOutput.linear.x = F_VEL;
        }
        else if (person_x < MIN_DEPTH) {
            RCLCPP_INFO(this->get_logger(), "BACKWARD");
            velOutput.linear.x = -F_VEL;
        }
        else {
            RCLCPP_INFO(this->get_logger(), "STOP");
            velOutput.linear.x = 0;
        }
        
        pub_->publish(velOutput);
    }

    void LostHuman() {
        velOutput.linear.x = 0;
        velOutput.angular.z = 0;
        pub_->publish(velOutput);
    }

    rclcpp::Client<moiro_interfaces::srv::TargetPose>::SharedPtr srv_client;
    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr pub_;
    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::Service<moiro_interfaces::srv::TargetDepth>::SharedPtr srv_depth;
    double MAX_DEPTH;
    double MIN_DEPTH;
    double MAX_Y;
    double MIN_Y;
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    auto turtlebot3_controller = std::make_shared<HumanFollower>();
    rclcpp::executors::SingleThreadedExecutor executor;
    executor.add_node(turtlebot3_controller);
    executor.spin();
    rclcpp::shutdown();
    return 0;
}