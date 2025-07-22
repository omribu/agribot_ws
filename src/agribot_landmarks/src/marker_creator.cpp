#include <opencv2/opencv.hpp>
#include <opencv2/objdetect/aruco_detector.hpp>
#include <opencv2/objdetect/aruco_dictionary.hpp>
#include <iostream>
#include <fstream>
#include <map>
#include <string>
#include <vector>
#include <sstream>
#include <chrono>
#include <iomanip>
#include <filesystem>


// This creates a Aruco Marker and saves it in the marker folder.
// After it is possible to add to each id marker in the cvs dictionary its location in world cooridainates and orientation. 
//     single marker : ros2 run agribot_landmarks marker_creator --create 0 
//     batch of markers : ros2 run agribot_landmarks marker_creator --batch 0 9  (creates 9 markers from id 0 to 9) 
//     single marker : ros2 run agribot_landmarks marker_creator --add 0 0.0 0.0 0.0 (add  to the csv dictionary id 0 , x, y, z) 





// Structure to hold marker information
struct MarkerInfo {
    int id;
    double world_x, world_y, world_z;  // World coordinates in meters
    double roll, pitch, yaw;           // Orientation in radians
    double marker_size_cm;             // Physical size in cm
    std::string location_description;
    std::string created_date;
    
    MarkerInfo() : id(0), world_x(0), world_y(0), world_z(0), 
                   roll(0), pitch(0), yaw(0), marker_size_cm(40.0) {}
    
    MarkerInfo(int _id, double _x, double _y, double _z,
               double _roll = 0, double _pitch = 0, double _yaw = 0,
               double _size = 40.0, const std::string& _desc = "") 
        : id(_id), world_x(_x), world_y(_y), world_z(_z),
          roll(_roll), pitch(_pitch), yaw(_yaw), marker_size_cm(_size), 
          location_description(_desc) {}
};

// Marker Database Class
class MarkerDatabase {
private:
    std::map<int, MarkerInfo> markers;
    std::string database_file;
    
public:
    MarkerDatabase(const std::string& db_file = "/home/volcani/agribot_ws/src/agribot_landmarks/markers/marker_database.csv") 
        : database_file(db_file) {
        loadDatabase();
    }
    
    void addMarker(const MarkerInfo& marker) {
        markers[marker.id] = marker;
        saveDatabase();
        std::cout << "✓ Added marker ID " << marker.id << " at world position (" 
                  << marker.world_x << ", " << marker.world_y << ", " << marker.world_z << ") meters" << std::endl;
    }
    
    bool getMarker(int id, MarkerInfo& marker) {
        auto it = markers.find(id);
        if (it != markers.end()) {
            marker = it->second;
            return true;
        }
        return false;
    }
    
    void updateMarkerLocation(int id, double x, double y, double z, const std::string& description = "") {
        auto it = markers.find(id);
        if (it != markers.end()) {
            it->second.world_x = x;
            it->second.world_y = y;
            it->second.world_z = z;
            if (!description.empty()) {
                it->second.location_description = description;
            }
            saveDatabase();
            std::cout << "✓ Updated marker ID " << id << " location to (" << x << ", " << y << ", " << z << ")" << std::endl;
        } else {
            std::cout << "✗ Marker ID " << id << " not found in database" << std::endl;
        }
    }
    
    void listAllMarkers() {
        std::cout << "\n==================== MARKER DATABASE ====================" << std::endl;
        std::cout << "ID\tWorld Position (x,y,z) [m]\tSize[cm]\tDescription" << std::endl;
        std::cout << "--------------------------------------------------------" << std::endl;
        
        for (const auto& pair : markers) {
            const MarkerInfo& m = pair.second;
            std::cout << m.id << "\t(" << std::fixed << std::setprecision(2)
                      << m.world_x << ", " << m.world_y << ", " << m.world_z << ")\t\t"
                      << m.marker_size_cm << "\t" << m.location_description << std::endl;
        }
        std::cout << "=========================================================" << std::endl;
    }
    
    void saveDatabase() {
        std::ofstream file(database_file);
        if (!file.is_open()) {
            std::cerr << "Error: Could not open database file for writing: " << database_file << std::endl;
            return;
        }
        
        // Write header
        file << "ID,World_X,World_Y,World_Z,Roll,Pitch,Yaw,Size_CM,Description,Created_Date\n";
        
        // Write marker data
        for (const auto& pair : markers) {
            const MarkerInfo& m = pair.second;
            file << m.id << "," << m.world_x << "," << m.world_y << "," << m.world_z << ","
                 << m.roll << "," << m.pitch << "," << m.yaw << ","
                 << m.marker_size_cm << ",\"" << m.location_description << "\",\"" 
                 << m.created_date << "\"\n";
        }
        
        file.close();
    }
    
    void loadDatabase() {
        std::ifstream file(database_file);
        if (!file.is_open()) {
            std::cout << "Database file not found, creating new database: " << database_file << std::endl;
            return;
        }
        
        std::string line;
        std::getline(file, line); // Skip header
        
        while (std::getline(file, line)) {
            if (line.empty()) continue;
            
            // Parse CSV line
            std::vector<std::string> fields;
            std::stringstream ss(line);
            std::string field;
            
            while (std::getline(ss, field, ',')) {
                // Remove quotes if present
                if (field.front() == '"' && field.back() == '"') {
                    field = field.substr(1, field.length() - 2);
                }
                fields.push_back(field);
            }
            
            if (fields.size() >= 8) {
                MarkerInfo marker;
                marker.id = std::stoi(fields[0]);
                marker.world_x = std::stod(fields[1]);
                marker.world_y = std::stod(fields[2]);
                marker.world_z = std::stod(fields[3]);
                marker.roll = std::stod(fields[4]);
                marker.pitch = std::stod(fields[5]);
                marker.yaw = std::stod(fields[6]);
                marker.marker_size_cm = std::stod(fields[7]);
                if (fields.size() > 8) marker.location_description = fields[8];
                if (fields.size() > 9) marker.created_date = fields[9];
                
                markers[marker.id] = marker;
            }
        }
        
        file.close();
        std::cout << "✓ Loaded " << markers.size() << " markers from database" << std::endl;
    }
    
    std::string getCurrentTimestamp() {
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        std::stringstream ss;
        ss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
        return ss.str();
    }
    
    void exportForDetection(const std::string& export_file = "/home/volcani/agribot_ws/src/agribot_landmarks/markers/detection_database.txt") {
        std::ofstream file(export_file);
        if (!file.is_open()) {
            std::cerr << "Error: Could not open export file: " << export_file << std::endl;
            return;
        }
        
        file << "# ArUco Marker Detection Database\n";
        file << "# Format: ID X Y Z ROLL PITCH YAW SIZE_CM DESCRIPTION\n";
        
        for (const auto& pair : markers) {
            const MarkerInfo& m = pair.second;
            file << m.id << " " << m.world_x << " " << m.world_y << " " << m.world_z << " "
                 << m.roll << " " << m.pitch << " " << m.yaw << " " << m.marker_size_cm 
                 << " " << m.location_description << "\n";
        }
        
        file.close();
        std::cout << "✓ Exported detection database to: " << export_file << std::endl;
    }
};

// Function to ensure directory exists
bool ensureDirectoryExists(const std::string& path) {
    try {
        if (!std::filesystem::exists(path)) {
            std::filesystem::create_directories(path);
            std::cout << "✓ Created directory: " << path << std::endl;
        }
        return true;
    } catch (const std::filesystem::filesystem_error& e) {
        std::cerr << "✗ Error creating directory: " << e.what() << std::endl;
        return false;
    }
}

// Function to create 40cm ArUco marker
bool createMarker(int marker_id, const std::string& output_dir) {
    // Calculate size for 40cm at 300 DPI
    // 40cm = 15.75 inches, 15.75 * 300 = 4724 pixels
    int marker_size_pixels = 4724;
    
    try {
        // Create 6x6 ArUco dictionary
        cv::aruco::Dictionary dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
        
        // Generate marker image
        cv::Mat marker_image;
        cv::aruco::generateImageMarker(dictionary, marker_id, marker_size_pixels, marker_image, 1);
        
        // Create output filename
        std::string output_path = output_dir + "/marker_" + std::to_string(marker_id) + "_40cm.png";
        
        // Save marker image
        if (cv::imwrite(output_path, marker_image)) {
            std::cout << "✓ Created marker ID " << marker_id << " -> " << output_path << std::endl;
            std::cout << "  Size: " << marker_size_pixels << "x" << marker_size_pixels << " pixels (40cm @ 300 DPI)" << std::endl;
            return true;
        } else {
            std::cerr << "✗ Failed to save marker image: " << output_path << std::endl;
            return false;
        }
        
    } catch (const cv::Exception& e) {
        std::cerr << "✗ OpenCV Error: " << e.what() << std::endl;
        return false;
    }
}

void printUsage(const std::string& program_name) {
    std::cout << "=== 40cm ArUco Marker Creator with World Coordinate Database ===" << std::endl;
    std::cout << "Usage: " << program_name << " [OPTIONS]" << std::endl;
    std::cout << std::endl;
    std::cout << "OPTIONS:" << std::endl;
    std::cout << "  -create <id>                    Create single marker with given ID" << std::endl;
    std::cout << "  -batch <start> <end>            Create batch of markers from start to end ID" << std::endl;
    std::cout << "  -add <id> <x> <y> <z> [desc]    Add marker location to database" << std::endl;
    std::cout << "  -update <id> <x> <y> <z> [desc] Update marker location in database" << std::endl;
    std::cout << "  -list                           List all markers in database" << std::endl;
    std::cout << "  -export                         Export database for detection use" << std::endl;
    std::cout << "  -help                           Show this help" << std::endl;
    std::cout << std::endl;
    std::cout << "EXAMPLES:" << std::endl;
    std::cout << "  " << program_name << " -create 0" << std::endl;
    std::cout << "  " << program_name << " -batch 0 9" << std::endl;
    std::cout << "  " << program_name << " -add 0 0.0 0.0 0.0 \"Origin marker\"" << std::endl;
    std::cout << "  " << program_name << " -add 1 5.0 0.0 0.0 \"5m east of origin\"" << std::endl;
    std::cout << "  " << program_name << " -update 1 5.2 0.1 0.0 \"Adjusted position\"" << std::endl;
    std::cout << "  " << program_name << " -list" << std::endl;
    std::cout << "  " << program_name << " -export" << std::endl;
    std::cout << std::endl;
    std::cout << "NOTES:" << std::endl;
    std::cout << "  - All markers are 6x6 ArUco DICT_6X6_250 format" << std::endl;
    std::cout << "  - Physical size: 40cm x 40cm" << std::endl;
    std::cout << "  - Print at 300 DPI for correct size" << std::endl;
    std::cout << "  - World coordinates in meters" << std::endl;
    std::cout << "  - Saved to: /home/volcani/agribot_ws/src/agribot_landmarks/markers/" << std::endl;
    std::cout << "=================================================================" << std::endl;
}

int main(int argc, char** argv)
{
    const std::string markers_dir = "/home/volcani/agribot_ws/src/agribot_landmarks/markers";
    
    if (argc < 2) {
        printUsage(argv[0]);
        return 1;
    }
    
    // Ensure markers directory exists
    if (!ensureDirectoryExists(markers_dir)) {
        std::cerr << "✗ Failed to create markers directory" << std::endl;
        return 1;
    }
    
    // Initialize database
    MarkerDatabase db;
    
    std::string command = argv[1];
    
    if (command == "-create" && argc >= 3) {
        int marker_id = std::stoi(argv[2]);
        
        if (createMarker(marker_id, markers_dir)) {
            // Add to database with default location (0,0,0)
            MarkerInfo marker(marker_id, 0.0, 0.0, 0.0, 0, 0, 0, 40.0, "");
            marker.created_date = db.getCurrentTimestamp();
            db.addMarker(marker);
            
            std::cout << "✓ Marker " << marker_id << " created and added to database" << std::endl;
            std::cout << "  Use -update to set world coordinates" << std::endl;
        }
        
    } else if (command == "-batch" && argc >= 4) {
        int start_id = std::stoi(argv[2]);
        int end_id = std::stoi(argv[3]);
        
        std::cout << "Creating batch of markers from " << start_id << " to " << end_id << "..." << std::endl;
        
        for (int i = start_id; i <= end_id; i++) {
            if (createMarker(i, markers_dir)) {
                MarkerInfo marker(i, 0.0, 0.0, 0.0, 0, 0, 0, 40.0, "");
                marker.created_date = db.getCurrentTimestamp();
                db.addMarker(marker);
            }
        }
        
        std::cout << "✓ Batch creation complete!" << std::endl;
        
    } else if (command == "-add" && argc >= 6) {
        int marker_id = std::stoi(argv[2]);
        double x = std::stod(argv[3]);
        double y = std::stod(argv[4]);
        double z = std::stod(argv[5]);
        std::string description = (argc > 6) ? argv[6] : "";
        
        MarkerInfo marker(marker_id, x, y, z, 0, 0, 0, 40.0, description);
        marker.created_date = db.getCurrentTimestamp();
        db.addMarker(marker);
        
    } else if (command == "-update" && argc >= 6) {
        int marker_id = std::stoi(argv[2]);
        double x = std::stod(argv[3]);
        double y = std::stod(argv[4]);
        double z = std::stod(argv[5]);
        std::string description = (argc > 6) ? argv[6] : "";
        
        db.updateMarkerLocation(marker_id, x, y, z, description);
        
    } else if (command == "-list") {
        db.listAllMarkers();
        
    } else if (command == "-export") {
        db.exportForDetection();
        
    } else if (command == "-help") {
        printUsage(argv[0]);
        
    } else {
        std::cerr << "✗ Invalid command or insufficient arguments" << std::endl;
        printUsage(argv[0]);
        return 1;
    }
    
    return 0;
}




