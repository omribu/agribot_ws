#!/usr/bin/env python3

import cv2
import numpy as np
import os
import csv
import argparse
import sys
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, Dict

@dataclass
class MarkerInfo:
    """Structure to hold marker information"""
    id: int
    world_x: float = 0.0
    world_y: float = 0.0
    world_z: float = 0.0
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    marker_size_cm: float = 40.0
    location_description: str = ""
    created_date: str = ""

class MarkerDatabase:
    """Marker Database Class"""
    
    def __init__(self, db_file: str = None):
        if db_file is None:
            base_path = os.path.expanduser("~/agribot_ws/src/agribot_landmarks/markers")
            db_file = os.path.join(base_path, "marker_database.csv")
        
        self.database_file = db_file
        self.markers: Dict[int, MarkerInfo] = {}
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.database_file), exist_ok=True)
        self.load_database()
    
    def add_marker(self, marker: MarkerInfo):
        """Add a marker to the database"""
        self.markers[marker.id] = marker
        self.save_database()
        print(f"✓ Added marker ID {marker.id} at world position "
              f"({marker.world_x}, {marker.world_y}, {marker.world_z}) meters")
    
    def get_marker(self, marker_id: int) -> Optional[MarkerInfo]:
        """Get a marker from the database"""
        return self.markers.get(marker_id)
    
    def update_marker_location(self, marker_id: int, x: float, y: float, z: float, 
                             description: str = ""):
        """Update marker location in the database"""
        if marker_id in self.markers:
            self.markers[marker_id].world_x = x
            self.markers[marker_id].world_y = y
            self.markers[marker_id].world_z = z
            if description:
                self.markers[marker_id].location_description = description
            self.save_database()
            print(f"✓ Updated marker ID {marker_id} location to ({x}, {y}, {z})")
        else:
            print(f"✗ Marker ID {marker_id} not found in database")
    
    def list_all_markers(self):
        """List all markers in the database"""
        print("\n" + "=" * 58)
        print("                    MARKER DATABASE")
        print("=" * 58)
        print("ID\tWorld Position (x,y,z) [m]\tSize[cm]\tDescription")
        print("-" * 58)
        
        for marker in self.markers.values():
            print(f"{marker.id}\t({marker.world_x:.2f}, {marker.world_y:.2f}, "
                  f"{marker.world_z:.2f})\t\t{marker.marker_size_cm}\t{marker.location_description}")
        
        print("=" * 58)
    
    def save_database(self):
        """Save the database to CSV file"""
        try:
            with open(self.database_file, 'w', newline='') as csvfile:
                fieldnames = ['ID', 'World_X', 'World_Y', 'World_Z', 'Roll', 'Pitch', 
                            'Yaw', 'Size_CM', 'Description', 'Created_Date']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for marker in self.markers.values():
                    writer.writerow({
                        'ID': marker.id,
                        'World_X': marker.world_x,
                        'World_Y': marker.world_y,
                        'World_Z': marker.world_z,
                        'Roll': marker.roll,
                        'Pitch': marker.pitch,
                        'Yaw': marker.yaw,
                        'Size_CM': marker.marker_size_cm,
                        'Description': marker.location_description,
                        'Created_Date': marker.created_date
                    })
        except Exception as e:
            print(f"Error saving database: {e}")
    
    def load_database(self):
        """Load the database from CSV file"""
        if not os.path.exists(self.database_file):
            print(f"Database file not found, creating new database: {self.database_file}")
            return
        
        try:
            with open(self.database_file, 'r') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    marker = MarkerInfo(
                        id=int(row['ID']),
                        world_x=float(row.get('World_X', 0.0)),
                        world_y=float(row.get('World_Y', 0.0)),
                        world_z=float(row.get('World_Z', 0.0)),
                        roll=float(row.get('Roll', 0.0)),
                        pitch=float(row.get('Pitch', 0.0)),
                        yaw=float(row.get('Yaw', 0.0)),
                        marker_size_cm=float(row.get('Size_CM', 40.0)),
                        location_description=row.get('Description', ''),
                        created_date=row.get('Created_Date', '')
                    )
                    self.markers[marker.id] = marker
            
            print(f"Loaded {len(self.markers)} markers from database")
        except Exception as e:
            print(f"Error loading database: {e}")
    
    def get_current_timestamp(self) -> str:
        """Get current timestamp as string"""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def export_for_detection(self, export_file: str = None):
        """Export database for detection use"""
        if export_file is None:
            base_path = os.path.expanduser("~/agribot_ws/src/agribot_landmarks/markers")
            export_file = os.path.join(base_path, "detection_database.txt")
        
        try:
            with open(export_file, 'w') as f:
                f.write("# ArUco Marker Detection Database\n")
                f.write("# Format: ID X Y Z ROLL PITCH YAW SIZE_CM DESCRIPTION\n")
                
                for marker in self.markers.values():
                    f.write(f"{marker.id} {marker.world_x} {marker.world_y} "
                           f"{marker.world_z} {marker.roll} {marker.pitch} "
                           f"{marker.yaw} {marker.marker_size_cm} "
                           f"{marker.location_description}\n")
            
            print(f" Exported detection database to: {export_file}")
        except Exception as e:
            print(f"Error exporting database: {e}")

def get_aruco_dict(dict_type: str = "DICT_6X6_250"):
    """ Get ArUco dictionary """
    try:
        # For OpenCV 4.5.4 and earlier
        if hasattr(cv2.aruco, dict_type):
            dict_id = getattr(cv2.aruco, dict_type)
            return cv2.aruco.Dictionary_get(dict_id)
        else:
            print(f"Dictionary {dict_type} not found")
            return None
    except AttributeError:
        try:
            # Fallback for newer OpenCV versions
            dict_id = getattr(cv2.aruco, dict_type)
            return cv2.aruco.getPredefinedDictionary(dict_id)
        except Exception as e:
            print(f"Could not create ArUco dictionary: {e}")
            return None

def generate_marker_image(dictionary, marker_id: int, size_px: int = 4724):
    """Generate marker image - compatible with OpenCV 4.5.4"""
    try:
        if hasattr(cv2.aruco, 'drawMarker'):
            marker_img = cv2.aruco.drawMarker(dictionary, marker_id, size_px)
            return marker_img
        else:
            # Alternative method for some versions
            marker_img = np.zeros((size_px, size_px), dtype=np.uint8)
            cv2.aruco.drawMarker(dictionary, marker_id, size_px, marker_img, 1)
            return marker_img
    except AttributeError:
        try:
            # For newer OpenCV versions (fallback)
            marker_img = cv2.aruco.generateImageMarker(dictionary, marker_id, size_px)
            return marker_img
        except Exception as e:
            print(f"Could not generate marker image: {e}")
            return None

def create_marker(marker_id: int, output_dir: str) -> bool:
    """Create 40cm ArUco marker"""
    # Calculate size for 40cm at 300 DPI
    # 40cm = 15.75 inches, 15.75 * 300 = 4724 pixels
    marker_size_pixels = 4724
    
    try:
        # Create 6x6 ArUco dictionary
        dictionary = get_aruco_dict("DICT_6X6_250")
        if dictionary is None:
            print("Failed to get ArUco dictionary")
            return False
        
        # Generate marker image
        marker_image = generate_marker_image(dictionary, marker_id, marker_size_pixels)
        if marker_image is None:
            print(f"Failed to generate marker {marker_id}")
            return False
        
        # Create output filename
        output_path = os.path.join(output_dir, f"marker_{marker_id}_40cm.png")
        
        # Save marker image
        if cv2.imwrite(output_path, marker_image):
            print(f"Created marker ID {marker_id} -> {output_path}")
            print(f"  Size: {marker_size_pixels}x{marker_size_pixels} pixels (40cm @ 300 DPI)")
            return True
        else:
            print(f"Failed to save marker image: {output_path}")
            return False
            
    except Exception as e:
        print(f"Error creating marker {marker_id}: {e}")
        return False

def print_usage(program_name: str):
    """Print usage information"""
    print("=== 40cm ArUco Marker Creator with World Coordinate Database ===")
    print(f"Usage: {program_name} [OPTIONS]")
    print()
    print("OPTIONS:")
    print("  -create <id>                    Create single marker with given ID")
    print("  -batch <start> <end>            Create batch of markers from start to end ID")
    print("  -add <id> <x> <y> <z> [desc]    Add marker location to database")
    print("  -update <id> <x> <y> <z> [desc] Update marker location in database")
    print("  -list                           List all markers in database")
    print("  -export                         Export database for detection use")
    print("  -help                           Show this help")
    print()
    print("EXAMPLES:")
    print(f"  {program_name} -create 0")
    print(f"  {program_name} -batch 0 9")
    print(f"  {program_name} -add 0 0.0 0.0 0.0 \"Origin marker\"")
    print(f"  {program_name} -add 1 5.0 0.0 0.0 \"5m east of origin\"")
    print(f"  {program_name} -update 1 5.2 0.1 0.0 \"Adjusted position\"")
    print(f"  {program_name} -list")
    print(f"  {program_name} -export")
    print()
    print("NOTES:")
    print("  - All markers are 6x6 ArUco DICT_6X6_250 format")
    print("  - Physical size: 40cm x 40cm")
    print("  - Print at 300 DPI for correct size")
    print("  - World coordinates in meters")
    print("  - Saved to: ~/agribot_ws/src/agribot_landmarks/markers/")
    print("=" * 65)

def main():

    markers_dir = os.path.expanduser("~/agribot_ws/src/agribot_landmarks/markers")
    
    all_args = sys.argv[1:] if len(sys.argv) > 1 else []
    
    if not all_args:
        print_usage(sys.argv[0])
        return 1
    
    os.makedirs(markers_dir, exist_ok=True)
    
    db = MarkerDatabase()
    
    command = all_args[0]
    
    try:
        if command == "-create" and len(all_args) >= 2:
            marker_id = int(all_args[1])
            
            if create_marker(marker_id, markers_dir):
            
                marker = MarkerInfo(
                    id=marker_id,
                    world_x=0.0,
                    world_y=0.0,
                    world_z=0.0,
                    roll=0.0,
                    pitch=0.0,
                    yaw=0.0,
                    marker_size_cm=40.0,
                    location_description="",
                    created_date=db.get_current_timestamp()
                )
                db.add_marker(marker)
                
                print(f"Marker {marker_id} created and added to database")
                print("  Use -update to set world coordinates")
        
        elif command == "-batch" and len(all_args) >= 3:
            start_id = int(all_args[1])
            end_id = int(all_args[2])
            
            print(f"Creating batch of markers from {start_id} to {end_id}...")
            
            for i in range(start_id, end_id + 1):
                if create_marker(i, markers_dir):
                    marker = MarkerInfo(
                        id=i,
                        world_x=0.0,
                        world_y=0.0,
                        world_z=0.0,
                        roll=0.0,
                        pitch=0.0,
                        yaw=0.0,
                        marker_size_cm=40.0,
                        location_description="",
                        created_date=db.get_current_timestamp()
                    )
                    db.add_marker(marker)
            
            print("Batch creation complete")
        
        elif command == "-add" and len(all_args) >= 5:
            marker_id = int(all_args[1])
            x = float(all_args[2])
            y = float(all_args[3])
            z = float(all_args[4])
            description = all_args[5] if len(all_args) > 5 else ""
            
            marker = MarkerInfo(
                id=marker_id,
                world_x=x,
                world_y=y,
                world_z=z,
                roll=0.0,
                pitch=0.0,
                yaw=0.0,
                marker_size_cm=40.0,
                location_description=description,
                created_date=db.get_current_timestamp()
            )
            db.add_marker(marker)
        
        elif command == "-update" and len(all_args) >= 5:
            marker_id = int(all_args[1])
            x = float(all_args[2])
            y = float(all_args[3])
            z = float(all_args[4])
            description = all_args[5] if len(all_args) > 5 else ""
            
            db.update_marker_location(marker_id, x, y, z, description)
        
        elif command == "-list":
            db.list_all_markers()
        
        elif command == "-export":
            db.export_for_detection()
        
        elif command == "-help":
            print_usage(sys.argv[0])
        
        else:
            print("Invalid command or insufficient arguments")
            print_usage(sys.argv[0])
            return 1
    
    except ValueError as e:
        print(f"Invalid argument: {e}")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())