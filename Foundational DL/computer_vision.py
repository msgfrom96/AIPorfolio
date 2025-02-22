import cv2
import time
import os
import urllib.request

def download_file(url, local_filename):
    """Helper function to download missing model files"""
    try:
        print(f"Downloading {local_filename} from {url}...")
        urllib.request.urlretrieve(url, local_filename)
        print("Download completed successfully!")
        return True
    except Exception as e:
        print(f"Failed to download {local_filename}: {str(e)}")
        return False

# Verify and download missing model files if needed
model_files = {
    "deploy.prototxt": "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
    "res10_300x300_ssd_iter_140000.caffemodel": "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
}

for file_name, url in model_files.items():
    if not os.path.exists(file_name):
        if not download_file(url, file_name):
            print("Failed to download required model files. Exiting...")
            exit(1)

try:
    # Load pre-trained face detection model
    face_net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel")
except Exception as e:
    print(f"Failed to load model: {str(e)}")
    exit(1)

# Initialize webcam
video_capture = cv2.VideoCapture(0)
if not video_capture.isOpened():
    print("Error: Could not open video source")
    exit(1)

# Initialize FPS calculation
fps = 0
frame_count = 0
start_time = time.time()

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    if not ret:
        print("Error: Failed to capture frame")
        break

    # Resize frame for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    try:
        # Face detection using DNN
        (h, w) = small_frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(small_frame, (300, 300)), 1.0, 
                                    (300, 300), (104.0, 177.0, 123.0))
        face_net.setInput(blob)
        detections = face_net.forward()

        # Process detections
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.7:
                box = [
                    detections[0, 0, i, 3] * w,
                    detections[0, 0, i, 4] * h,
                    detections[0, 0, i, 5] * w,
                    detections[0, 0, i, 6] * h
                ]
                (startX, startY, endX, endY) = [int(coord) for coord in box]
                
                # Expand face coordinates to original frame size
                top = startY * 4
                right = endX * 4
                bottom = endY * 4
                left = startX * 4

                # Draw bounding box and label
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.putText(frame, "Person", (left + 6, bottom - 6), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    except Exception as e:
        print(f"Error during face detection: {str(e)}")
        break

    # Calculate FPS
    frame_count += 1
    if frame_count >= 10:
        fps = frame_count / (time.time() - start_time)
        frame_count = 0
        start_time = time.time()
    
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 20),
              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display result
    cv2.imshow('Video', frame)

    # Exit on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
video_capture.release()
cv2.destroyAllWindows()
