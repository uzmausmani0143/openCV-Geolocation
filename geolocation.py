import streamlit as st
import cv2
import numpy as np
import geocoder

st.title('OpenCV Streamlit App with Geolocation')

# Fetch geolocation once to avoid repeated API calls
g = geocoder.ip('me')  # 'me' uses the public IP of the machine

# If the geolocation is incorrect, you can override it manually
if g.city == "Delhi" and not g.latlng == [28.7041, 77.1025]:  # Example of checking if the location is wrong
    # Set a manual location (Bhubaneswar example)
    location_info = "Location: Bhubaneswar, Odisha, India - Lat: 20.2961, Lng: 85.8245"
else:
    # If the location is accurate, use the fetched geolocation
    location_info = f"Location: {g.city}, {g.state}, {g.country} - Lat: {g.latlng[0]}, Lng: {g.latlng[1]}"

# URL of the external camera stream
video_url = 'http://192.168.137.156:8080/video'
cap = cv2.VideoCapture(video_url)

if not cap.isOpened():
    st.error("Error: Could not open video stream")
else:
    st.text("Video stream opened successfully")

# Create a placeholder to display the video
frame_placeholder = st.empty()

# Add a stop button
stop_button = st.button('Stop')

while True:
    ret, frame = cap.read()
    if not ret:
        st.text("Failed to grab frame")
        break

    # Display the location on the frame
    cv2.putText(frame, location_info, (10, 50), cv2.FONT_HERSHEY_COMPLEX , 1.5, (0, 0, 0), 3, cv2.LINE_AA)
    
    # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Display the frame in the Streamlit app
    frame_placeholder.image(frame_rgb, channels='RGB')

    # Check if the stop button was clicked
    if stop_button:
        break

# Release the capture
cap.release()
