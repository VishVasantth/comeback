import os
import requests
import sys

def download_test_data():
    """Download test images for obstacle detection"""
    test_dir = "test_data"
    os.makedirs(test_dir, exist_ok=True)
    
    test_images = [
        {
            "url": "https://media.istockphoto.com/id/1212275972/photo/group-of-people-walking-down-on-the-street.jpg?s=612x612&w=0&k=20&c=yVQgm6qvSQQM9DFLLVsnqMtZMJmc_Q5L5R6qaqHVtQw=",
            "filename": "people_walking.jpg",
            "description": "People walking on street (obstacle test)"
        },
        {
            "url": "https://media.istockphoto.com/id/1333176517/photo/pothole-on-the-road-dangerous-hole-in-the-asphalt.jpg?s=612x612&w=0&k=20&c=pMVJAmwyBg6mc0z7R5sCzVPGa6SIKk24Kg_bsaZ7agI=",
            "filename": "pothole.jpg",
            "description": "Pothole in road (obstacle test)"
        },
        {
            "url": "https://media.istockphoto.com/id/1159397365/photo/car-driving-on-flooded-road-street-after-heavy-rain.jpg?s=612x612&w=0&k=20&c=ZjmJg5xyJWY62ITsGQgUl5I_QCVczm9Jw-bNF9B4L-8=",
            "filename": "flooded_road.jpg",
            "description": "Flooded road (obstacle test)"
        },
        {
            "url": "https://media.istockphoto.com/id/480814292/photo/campus-statue.jpg?s=612x612&w=0&k=20&c=mJQHgGuvxdV-1-t72V9ydNKGMOPmuX8RbdRFMR8w93k=",
            "filename": "campus_statue.jpg",
            "description": "Campus statue (landmark test)"
        },
        {
            "url": "https://media.istockphoto.com/id/1435964963/photo/campus-of-college-university-education-building-with-nature-greenery-sunny-clear-blue-sky-day.jpg?s=612x612&w=0&k=20&c=hEvv4fZ2VLJI3h1pKW0jKhAq0UH6_0n8rS_Jt7XrQn0=",
            "filename": "campus_path.jpg",
            "description": "Campus path (path test)"
        }
    ]
    
    for image in test_images:
        filename = os.path.join(test_dir, image["filename"])
        
        if os.path.exists(filename):
            print(f"File {filename} already exists. Skipping.")
            continue
        
        print(f"Downloading {image['description']} to {filename}...")
        try:
            response = requests.get(image["url"], stream=True)
            response.raise_for_status()
            
            with open(filename, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print(f"Successfully downloaded {filename}")
        except Exception as e:
            print(f"Error downloading {filename}: {e}")
    
    # Create a test pattern with artificial obstacles
    try:
        import cv2
        import numpy as np
        
        print("Creating artificial obstacle test pattern...")
        obstacle_img = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Add a road/path
        cv2.rectangle(obstacle_img, (0, 180), (640, 300), (80, 80, 80), -1)  # Gray road
        
        # Add road markings
        for i in range(10):
            cv2.rectangle(obstacle_img, (i*80, 235), (i*80+30, 245), (255, 255, 255), -1)
        
        # Add a person-like obstacle
        cv2.rectangle(obstacle_img, (320, 200), (350, 280), (0, 0, 255), -1)  # Red box for person
        
        # Add labels
        cv2.putText(obstacle_img, "Test Obstacle", (290, 190), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Save the image
        obstacle_path = os.path.join(test_dir, "synthetic_obstacle.jpg")
        cv2.imwrite(obstacle_path, obstacle_img)
        print(f"Created synthetic obstacle image: {obstacle_path}")
    except ImportError:
        print("OpenCV not available. Skipping synthetic obstacle creation.")
    
    print("\nTest data setup complete. You can now use these test images with the object detection system.")
    print("To use a test image, set the RTSP_URL environment variable to the path of a test image.")
    print("Example:")
    print("export RTSP_URL=test_data/people_walking.jpg")

if __name__ == "__main__":
    download_test_data() 