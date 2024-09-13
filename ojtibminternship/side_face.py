import cv2
import face_recognition
import os
from datetime import datetime
import geocoder
from deepface import DeepFace
import dlib

# Initialize face recognition
known_face_encodings = []
known_face_names = []

# Store unknown face encodings to prevent multiple captures
captured_unknown_face_encodings = []

# Directory containing images of known persons
known_faces_dir = r"C:\Users\Acer\Downloads\All_Work\ojtibminternship\known"

# Directory to save images of unknown persons
unknown_faces_dir = r"C:\Users\Acer\Downloads\All_Work\ojtibminternship\unknown"

# Load each image file and extract face encodings (support for side profiles)
for image_name in os.listdir(known_faces_dir):
    image_path = os.path.join(known_faces_dir, image_name)
    image = face_recognition.load_image_file(image_path)
    face_encodings = face_recognition.face_encodings(image)

    if face_encodings:
        face_encoding = face_encodings[0]
        # Extract the base name without view suffix (e.g., Uzma_side.jpg -> Uzma)
        name = os.path.splitext(image_name)[0].split("_")[0]
        known_face_encodings.append(face_encoding)
        known_face_names.append(name)

# Fetch geolocation once to avoid repeated API calls
g = geocoder.ip('me')

# Set a manual location if geolocation is incorrect
if g.city == "Bhubaneswar" and not g.latlng == [28.7041, 77.1025]:
    location_info = "Location: Bhubaneswar, Odisha, India - Lat: 20.2961, Lng: 85.8245"
else:
    location_info = f"Location: {g.city}, {g.state}, {g.country} - Lat: {g.latlng[0]}, Lng: {g.latlng[1]}"

# Initialize webcam
video_capture = cv2.VideoCapture(0)

# Initialize dlib's CNN-based face detector
cnn_model_path = r"C:\Users\Acer\Downloads\All_Work\ojtibminternship\mmod_human_face_detector.dat"
cnn_face_detector = dlib.cnn_face_detection_model_v1(cnn_model_path)  # Ensure you have this file

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # Convert the image from BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Find all the faces and face encodings in the current frame (using face_recognition for frontal faces)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Use Dlib's CNN model to detect side and frontal faces
    dlib_face_detections = cnn_face_detector(rgb_frame, 1)
    
    # Adding Dlib detected faces to face_locations if any
    for detection in dlib_face_detections:
        rect = detection.rect
        top, right, bottom, left = rect.top(), rect.right(), rect.bottom(), rect.left()
        # Adding dlib face detections to face_recognition locations
        face_locations.append((top, right, bottom, left))
        face_encodings += face_recognition.face_encodings(rgb_frame, [(top, right, bottom, left)])

    # Perform emotion detection using DeepFace
    try:
        faces = DeepFace.analyze(img_path=rgb_frame, actions=['emotion'], enforce_detection=False)
    except Exception as e:
        print(f"Emotion detection error: {e}")
        faces = []

    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = face_distances.argmin()

        if matches[best_match_index]:
            name = known_face_names[best_match_index]
        else:
            name = "Unknown"
            # Capture unknown faces but no alarm
            if not any(face_recognition.compare_faces(captured_unknown_face_encodings, face_encoding, tolerance=0.6)):
                top, right, bottom, left = face_location
                face_image = frame[top:bottom, left:right]
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                unknown_image_path = os.path.join(unknown_faces_dir, f"unknown_{timestamp}.jpg")
                cv2.imwrite(unknown_image_path, face_image)
                captured_unknown_face_encodings.append(face_encoding)

        # Draw a box around the face
        top, right, bottom, left = face_location
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display emotions if detected
    for face in faces:
        x, y, w, h = face['region']['x'], face['region']['y'], face['region']['w'], face['region']['h']
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, face['dominant_emotion'], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (54, 219, 9), 2)

    # Display the location on the frame
    cv2.putText(frame, location_info, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (7, 36, 250), 2, cv2.LINE_AA)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Break the loop with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
video_capture.release()
cv2.destroyAllWindows()
