import cv2
import streamlit as st
import numpy as np
from deepface import DeepFace

# Analyze facial attributes
def analyze_frame(frame):
    result = DeepFace.analyze(
        img_path=frame,
        actions=['age', 'gender', 'race', 'emotion'],
        enforce_detection=False,
        detector_backend="opencv",
        align=True,
        silent=True
    )
    return result

# Overlay styled text on the video frame
def overlay_text_on_frame(frame, texts):
    overlay = frame.copy()
    alpha = 0.85
    bar_height = 120
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], bar_height), (255, 255, 255), -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    y0, dy = 25, 20
    for i, text in enumerate(texts):
        y = y0 + i * dy
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_DUPLEX, 0.55, (30, 30, 30), 1, cv2.LINE_AA)

    return frame

# Main facial sentiment analysis function
def facesentiment():
    cap = cv2.VideoCapture(0)
    stframe = st.image([], caption="Webcam Feed", use_column_width=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            st.warning("Unable to access webcam.")
            break

        result = analyze_frame(frame)
        face = result[0]["region"]
        x, y, w, h = face['x'], face['y'], face['w'], face['h']

        # Draw rectangle and emotion
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 180, 255), 2)
        cv2.putText(frame, result[0]['dominant_emotion'], (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Info Texts
        texts = [
            f"👤 Age: {result[0]['age']}",
            f"📷 Confidence: {round(result[0]['face_confidence'], 2)}",
            f"⚧ Gender: {result[0]['dominant_gender']} ({round(result[0]['gender'][result[0]['dominant_gender']], 2)})",
            f"🌍 Race: {result[0]['dominant_race']}",
            f"😊 Emotion: {result[0]['dominant_emotion']} ({round(result[0]['emotion'][result[0]['dominant_emotion']], 1)}%)"
        ]

        frame_with_overlay = overlay_text_on_frame(frame_rgb, texts)
        stframe.image(frame_with_overlay, channels="RGB")

    cap.release()
    cv2.destroyAllWindows()

# Streamlit App
def main():
    st.set_page_config(page_title="Face Emotion App", layout="centered", initial_sidebar_state="auto")
    st.markdown(
        "<h2 style='text-align: center; color: #4C5C68;'>🎥 Real-Time Face Emotion Recognition</h2>",
        unsafe_allow_html=True
    )

    menu = ["📸 Webcam Detection"]
    choice = st.sidebar.selectbox("Navigation", menu)

    if choice == "📸 Webcam Detection":
        st.markdown(
            """<div style="background-color:#4C5C68;padding:10px;border-radius:10px">
               <h4 style="color:white;text-align:center;">
               Real-time facial emotion recognition using OpenCV, DeepFace, and Streamlit.
               </h4></div><br>""",
            unsafe_allow_html=True
        )
        facesentiment()

if __name__ == "__main__":
    main()
