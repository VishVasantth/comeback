import os
import requests
import sys

def download_file(url, destination):
    """Download a file from a URL to the specified destination."""
    try:
        if os.path.exists(destination):
            print(f"Model already exists at {destination}")
            return True
        
        print(f"Downloading model from {url} to {destination}...")
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        
        # Download the file
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Get file size
        file_size = int(response.headers.get('content-length', 0))
        if file_size == 0:
            file_size = None
        
        # Download with progress bar
        downloaded = 0
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    # Show progress if file size is known
                    if file_size:
                        done = int(50 * downloaded / file_size)
                        sys.stdout.write(f"\r[{'=' * done}{' ' * (50-done)}] {downloaded/1024/1024:.2f}MB/{file_size/1024/1024:.2f}MB")
                        sys.stdout.flush()
        
        print("\nDownload complete!")
        return True
    except Exception as e:
        print(f"Error downloading model: {e}")
        return False

if __name__ == "__main__":
    model_url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov11n.pt"
    model_path = os.path.join("models", "yolov11n.pt")
    
    success = download_file(model_url, model_path)
    
    if success:
        abs_path = os.path.abspath(model_path)
        print(f"Model downloaded successfully to: {abs_path}")
        print("You can now start the detection service.")
    else:
        print("Failed to download the model. Please try again or download it manually.")
        print(f"Manual download URL: {model_url}")
        print(f"Save the file to: {os.path.abspath(model_path)}")