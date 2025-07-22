#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <std_msgs/msg/header.hpp>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/objdetect/aruco_detector.hpp>
#include <opencv2/objdetect/aruco_dictionary.hpp>
#include <opencv2/calib3d.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include <iostream>
#include <fstream>
#include <map>
#include <string>
#include <vector>
#include <sstream>
#include <chrono>
#include <memory>

// Custom message for ArUco detection results
struct ArucoDetection {
    int id;
    std::vector<cv::Point2f> corners;
    cv::Point2f center;
    double area;
    
    // World coordinate info
    bool has_world_coords;
    double world_x, world_y, world_z;
    std::string description;
    
    ArucoDetection() : has_world_coords(false), world_x(0), world_y(0), world_z(0) {}
};

// Member variables struct
struct MarkerWorldInfo {
    int id;
    double world_x, world_y, world_z;
    double roll, pitch, yaw;
    double size_cm;
    std::string description;
};

class ArucoDetectorNode : public rclcpp::Node
{
public:
    ArucoDetectorNode() : Node("aruco_detector")
    {
        // Declare parameters
        this->declare_parameter("camera_topic", "/camera/image_raw");
        this->declare_parameter("database_file", "/home/volcani/agribot_ws/src/agribot_landmarks/markers/marker_database.csv");
        this->declare_parameter("marker_size", 0.40); // 40cm in meters
        this->declare_parameter("camera_frame", "camera_link");
        this->declare_parameter("world_frame", "map");
        this->declare_parameter("publish_tf", true);
        this->declare_parameter("publish_markers", true);
        this->declare_parameter("detection_rate", 10.0); // Hz
        
        // Get parameters
        camera_topic_ = this->get_parameter("camera_topic").as_string();
        database_file_ = this->get_parameter("database_file").as_string();
        marker_size_ = this->get_parameter("marker_size").as_double();
        camera_frame_ = this->get_parameter("camera_frame").as_string();
        world_frame_ = this->get_parameter("world_frame").as_string();
        publish_tf_ = this->get_parameter("publish_tf").as_bool();
        publish_markers_ = this->get_parameter("publish_markers").as_bool();
        detection_rate_ = this->get_parameter("detection_rate").as_double();
        
        // Initialize ArUco detector
        initializeAruco();
        
        // Load marker database
        loadMarkerDatabase();
        
        // Initialize camera parameters (you should calibrate your camera)
        initializeCameraParameters();
        
        // Create subscribers
        image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            camera_topic_, 10, std::bind(&ArucoDetectorNode::imageCallback, this, std::placeholders::_1));
        
        // Create publishers
        detection_pub_ = this->create_publisher<geometry_msgs::msg::PoseArray>("/aruco_detections", 10);
        world_pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseArray>("/aruco_world_poses", 10);
        
        if (publish_markers_) {
            marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("/aruco_markers", 10);
        }
        
        // Create TF broadcaster
        if (publish_tf_) {
            tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
        }
        
        // Create timer for periodic database reload
        database_timer_ = this->create_wall_timer(
            std::chrono::seconds(30),
            std::bind(&ArucoDetectorNode::reloadDatabase, this));
        
        RCLCPP_INFO(this->get_logger(), "ArUco Detector Node initialized");
        RCLCPP_INFO(this->get_logger(), "Camera topic: %s", camera_topic_.c_str());
        RCLCPP_INFO(this->get_logger(), "Database file: %s", database_file_.c_str());
        RCLCPP_INFO(this->get_logger(), "Marker size: %.2f meters", marker_size_);
        RCLCPP_INFO(this->get_logger(), "Loaded %zu markers from database", marker_database_.size());
    }

private:
    void initializeAruco()
    {
        try {
            // Create 6x6 ArUco dictionary (same as your markers)
            dictionary_ = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
            
            // Create detector parameters
            detector_params_ = cv::aruco::DetectorParameters();
            
            // Tune parameters for better detection
            detector_params_.adaptiveThreshWinSizeMin = 3;
            detector_params_.adaptiveThreshWinSizeMax = 23;
            detector_params_.adaptiveThreshWinSizeStep = 10;
            detector_params_.adaptiveThreshConstant = 7;
            detector_params_.minMarkerPerimeterRate = 0.03;
            detector_params_.maxMarkerPerimeterRate = 4.0;
            detector_params_.polygonalApproxAccuracyRate = 0.03;
            detector_params_.minCornerDistanceRate = 0.05;
            detector_params_.minDistanceToBorder = 3;
            detector_params_.minMarkerDistanceRate = 0.05;
            detector_params_.cornerRefinementMethod = cv::aruco::CORNER_REFINE_SUBPIX;
            detector_params_.cornerRefinementWinSize = 5;
            detector_params_.cornerRefinementMaxIterations = 30;
            detector_params_.cornerRefinementMinAccuracy = 0.1;
            
            // Create detector
            detector_ = cv::aruco::ArucoDetector(dictionary_, detector_params_);
            
            RCLCPP_INFO(this->get_logger(), "ArUco detector initialized with DICT_6X6_250");
            
        } catch (const cv::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Failed to initialize ArUco detector: %s", e.what());
        }
    }
    
    void initializeCameraParameters()
    {
        // TODO: Replace with your actual camera calibration parameters
        // These are example parameters - you should calibrate your camera
        camera_matrix_ = (cv::Mat_<double>(3, 3) << 
            800, 0, 320,
            0, 800, 240,
            0, 0, 1);
        
        dist_coeffs_ = (cv::Mat_<double>(4, 1) << 0.1, -0.2, 0.0, 0.0);
        
        RCLCPP_WARN(this->get_logger(), "Using default camera parameters - please calibrate your camera!");
    }
    
    void loadMarkerDatabase()
    {
        marker_database_.clear();
        
        std::ifstream file(database_file_);
        if (!file.is_open()) {
            RCLCPP_ERROR(this->get_logger(), "Could not open marker database: %s", database_file_.c_str());
            return;
        }
        
        std::string line;
        std::getline(file, line); // Skip header
        
        while (std::getline(file, line)) {
            if (line.empty()) continue;
            
            std::vector<std::string> fields;
            std::stringstream ss(line);
            std::string field;
            
            while (std::getline(ss, field, ',')) {
                // Remove quotes if present
                if (!field.empty() && field.front() == '"' && field.back() == '"') {
                    field = field.substr(1, field.length() - 2);
                }
                fields.push_back(field);
            }
            
            if (fields.size() >= 8) {
                MarkerWorldInfo info;
                info.id = std::stoi(fields[0]);
                info.world_x = std::stod(fields[1]);
                info.world_y = std::stod(fields[2]);
                info.world_z = std::stod(fields[3]);
                info.roll = std::stod(fields[4]);
                info.pitch = std::stod(fields[5]);
                info.yaw = std::stod(fields[6]);
                info.size_cm = std::stod(fields[7]);
                if (fields.size() > 8) info.description = fields[8];
                
                marker_database_[info.id] = info;
            }
        }
        
        file.close();
        RCLCPP_INFO(this->get_logger(), "Loaded %zu markers from database", marker_database_.size());
    }
    
    void reloadDatabase()
    {
        loadMarkerDatabase();
    }
    
    void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        try {
            // Convert ROS image to OpenCV
            cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
            cv::Mat image = cv_ptr->image;
            
            // Detect ArUco markers
            std::vector<ArucoDetection> detections = detectMarkers(image);
            
            // Process detections
            if (!detections.empty()) {
                publishDetections(detections, msg->header);
                
                if (publish_markers_) {
                    publishVisualizationMarkers(detections, msg->header);
                }
                
                if (publish_tf_) {
                    publishTransforms(detections, msg->header);
                }
                
                // Log detection results
                RCLCPP_INFO(this->get_logger(), "Detected %zu markers", detections.size());
                for (const auto& detection : detections) {
                    if (detection.has_world_coords) {
                        RCLCPP_INFO(this->get_logger(), "  ID %d: World pos (%.2f, %.2f, %.2f) - %s",
                                   detection.id, detection.world_x, detection.world_y, detection.world_z,
                                   detection.description.c_str());
                    } else {
                        RCLCPP_INFO(this->get_logger(), "  ID %d: No world coordinates in database", detection.id);
                    }
                }
            }
            
        } catch (const cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "CV bridge exception: %s", e.what());
        } catch (const cv::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "OpenCV exception: %s", e.what());
        }
    }
    
    std::vector<ArucoDetection> detectMarkers(const cv::Mat& image)
    {
        std::vector<ArucoDetection> detections;
        
        // Detect markers
        std::vector<int> ids;
        std::vector<std::vector<cv::Point2f>> corners;
        std::vector<std::vector<cv::Point2f>> rejected;
        
        detector_.detectMarkers(image, corners, ids, rejected);
        
        if (ids.empty()) {
            return detections;
        }
        
        // Process each detected marker
        for (size_t i = 0; i < ids.size(); i++) {
            ArucoDetection detection;
            detection.id = ids[i];
            detection.corners = corners[i];
            
            // Calculate center point
            detection.center = cv::Point2f(0, 0);
            for (const auto& corner : corners[i]) {
                detection.center += corner;
            }
            detection.center /= 4.0f;
            
            // Calculate area
            detection.area = cv::contourArea(corners[i]);
            
            // Look up world coordinates
            auto it = marker_database_.find(detection.id);
            if (it != marker_database_.end()) {
                detection.has_world_coords = true;
                detection.world_x = it->second.world_x;
                detection.world_y = it->second.world_y;
                detection.world_z = it->second.world_z;
                detection.description = it->second.description;
            }
            
            detections.push_back(detection);
        }
        
        return detections;
    }
    
    void publishDetections(const std::vector<ArucoDetection>& detections, const std_msgs::msg::Header& header)
    {
        // Publish camera frame detections
        geometry_msgs::msg::PoseArray detection_msg;
        detection_msg.header = header;
        detection_msg.header.frame_id = camera_frame_;
        
        // Publish world frame poses
        geometry_msgs::msg::PoseArray world_pose_msg;
        world_pose_msg.header = header;
        world_pose_msg.header.frame_id = world_frame_;
        
        for (const auto& detection : detections) {
            // Camera frame pose (you could estimate pose using solvePnP)
            geometry_msgs::msg::Pose camera_pose;
            camera_pose.position.x = detection.center.x;
            camera_pose.position.y = detection.center.y;
            camera_pose.position.z = 0.0;
            camera_pose.orientation.w = 1.0;
            
            detection_msg.poses.push_back(camera_pose);
            
            // World frame pose
            if (detection.has_world_coords) {
                geometry_msgs::msg::Pose world_pose;
                world_pose.position.x = detection.world_x;
                world_pose.position.y = detection.world_y;
                world_pose.position.z = detection.world_z;
                world_pose.orientation.w = 1.0;
                
                world_pose_msg.poses.push_back(world_pose);
            }
        }
        
        detection_pub_->publish(detection_msg);
        world_pose_pub_->publish(world_pose_msg);
    }
    
    void publishVisualizationMarkers(const std::vector<ArucoDetection>& detections, const std_msgs::msg::Header& header)
    {
        visualization_msgs::msg::MarkerArray marker_array;
        
        for (size_t i = 0; i < detections.size(); i++) {
            const auto& detection = detections[i];
            
            if (!detection.has_world_coords) continue;
            
            visualization_msgs::msg::Marker marker;
            marker.header = header;
            marker.header.frame_id = world_frame_;
            marker.ns = "aruco_markers";
            marker.id = detection.id;
            marker.type = visualization_msgs::msg::Marker::CUBE;
            marker.action = visualization_msgs::msg::Marker::ADD;
            
            // Position
            marker.pose.position.x = detection.world_x;
            marker.pose.position.y = detection.world_y;
            marker.pose.position.z = detection.world_z;
            marker.pose.orientation.w = 1.0;
            
            // Scale (40cm markers)
            marker.scale.x = marker_size_;
            marker.scale.y = marker_size_;
            marker.scale.z = 0.01; // Thin marker
            
            // Color (green for detected markers)
            marker.color.r = 0.0;
            marker.color.g = 1.0;
            marker.color.b = 0.0;
            marker.color.a = 0.8;
            
            // Lifetime
            marker.lifetime = rclcpp::Duration::from_seconds(1.0);
            
            marker_array.markers.push_back(marker);
            
            // Add text label
            visualization_msgs::msg::Marker text_marker;
            text_marker.header = marker.header;
            text_marker.ns = "aruco_text";
            text_marker.id = detection.id;
            text_marker.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
            text_marker.action = visualization_msgs::msg::Marker::ADD;
            
            text_marker.pose.position.x = detection.world_x;
            text_marker.pose.position.y = detection.world_y;
            text_marker.pose.position.z = detection.world_z + 0.3;
            text_marker.pose.orientation.w = 1.0;
            
            text_marker.scale.z = 0.2;
            text_marker.color.r = 1.0;
            text_marker.color.g = 1.0;
            text_marker.color.b = 1.0;
            text_marker.color.a = 1.0;
            
            text_marker.text = "ID:" + std::to_string(detection.id) + "\n" + detection.description;
            text_marker.lifetime = rclcpp::Duration::from_seconds(1.0);
            
            marker_array.markers.push_back(text_marker);
        }
        
        marker_pub_->publish(marker_array);
    }
    
    void publishTransforms(const std::vector<ArucoDetection>& detections, const std_msgs::msg::Header& header)
    {
        for (const auto& detection : detections) {
            if (!detection.has_world_coords) continue;
            
            geometry_msgs::msg::TransformStamped transform;
            transform.header = header;
            transform.header.frame_id = world_frame_;
            transform.child_frame_id = "aruco_" + std::to_string(detection.id);
            
            transform.transform.translation.x = detection.world_x;
            transform.transform.translation.y = detection.world_y;
            transform.transform.translation.z = detection.world_z;
            transform.transform.rotation.w = 1.0;
            
            tf_broadcaster_->sendTransform(transform);
        }
    }

    // Member variables
    std::map<int, MarkerWorldInfo> marker_database_;
    
    // ROS2 components
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr detection_pub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr world_pose_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
    rclcpp::TimerBase::SharedPtr database_timer_;
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    
    // ArUco detection
    cv::aruco::Dictionary dictionary_;
    cv::aruco::DetectorParameters detector_params_;
    cv::aruco::ArucoDetector detector_;
    
    // Camera parameters
    cv::Mat camera_matrix_;
    cv::Mat dist_coeffs_;
    
    // Parameters
    std::string camera_topic_;
    std::string database_file_;
    double marker_size_;
    std::string camera_frame_;
    std::string world_frame_;
    bool publish_tf_;
    bool publish_markers_;
    double detection_rate_;
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<ArucoDetectorNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}