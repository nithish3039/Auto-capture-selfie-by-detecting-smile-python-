import cv2
import os
import time

# Load the pre-trained Haar cascade classifier for smile detection
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

# Start capturing video from the webcam
cap = cv2.VideoCapture(0)

# Get the current working directory
current_directory = os.getcwd()
print(f"Images will be saved in: {current_directory}")

# Initialize variables
last_saved_time = time.time()
save_interval = 5  # Save an image once every 5 seconds

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect smiles in the frame with optimized parameters for better speed and accuracy
    smiles = smile_cascade.detectMultiScale(
        gray,
        scaleFactor=1.8,
        minNeighbors=20,
        minSize=(25, 25),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # If a smile is detected, draw rectangles around the smiles
    if len(smiles) > 0:
        for (x, y, w, h) in smiles:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw green rectangle
            cv2.putText(frame, 'Smile Detected!', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Check if it's time to save the image (to avoid redundant saving)
        if time.time() - last_saved_time > save_interval:
            # Define the image file name and path
            image_path = os.path.join(current_directory, 'smile_capture.jpg')

            # Capture the image automatically after detecting the smile
            cv2.imwrite(image_path, frame)
            print(f"Smile captured and saved as: {image_path}")
            last_saved_time = time.time()  # Update the time of last save

    else:
        # If no smile detected, display 'Keep Smiling!'
        cv2.putText(frame, 'Keep Smiling!', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Show the frame with detection result
    cv2.imshow('Smile Detection', frame)

    # Break the loop on 'ESC' key press
    if cv2.waitKey(1) & 0xFF == 27:  # ASCII code for ESC key
        break

# Release the video capture and destroy all OpenCV windows
cap.release()
cv2.destroyAllWindows()