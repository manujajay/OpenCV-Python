# OpenCV-Python

This repository is a comprehensive guide to using OpenCV with Python for image and video processing tasks. It includes detailed examples and tutorials on operations such as edge detection, face recognition, and object tracking.

## Prerequisites

- Python 3.6 or higher
- OpenCV library

## Installation

Install OpenCV for Python using pip:

```bash
pip install opencv-python-headless
```

## Basic Usage Examples

### 1. Edge Detection

```python
import cv2
import numpy as np

# Load an image
image = cv2.imread('input.jpg', cv2.IMREAD_GRAYSCALE)

# Detect edges using Canny
edges = cv2.Canny(image, 100, 200)

# Display edges
cv2.imshow('Edges', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### 2. Face Recognition

```python
import cv2

# Load a pre-trained face detector model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Capture video from camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Draw rectangles around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Display the resulting frame
    cv2.imshow('Frame', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()
```

## Contributing

Contributions to improve examples, extend documentation, or add new features are welcome. Please fork the repository, create a feature branch, and submit a pull request.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
