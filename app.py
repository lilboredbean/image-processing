import cv2
import streamlit as st
import numpy as np
import av
from deepface import DeepFace
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

RTC_CONFIGURATION = RTCConfiguration(
    {
        "iceServers": [
            {"urls": ["stun:stun.relay.metered.ca:80"]},
            {
                "urls": ["turn:global.relay.metered.ca:80"],
                "username": "your_username",
                "credential": "your_credential",
            },
            {
                "urls": ["turn:global.relay.metered.ca:443"],
                "username": "your_username",
                "credential": "your_credential",
            },
        ]
    }
)


def analyze_frame(frame):
    result = DeepFace.analyze(
        img_path=frame,
        actions=['age', 'gender', 'race', 'emotion'],
        enforce_detection=False,
        detector_backend="opencv",
        align=True,
        silent=True,
    )
    return result


def overlay_text_on_frame(frame, texts):
    overlay = frame.copy()
    alpha = 0.9
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 100), (255, 255, 255), -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    text_position = 15
    for text in texts:
        cv2.putText(frame, text, (10, text_position), cv2.FONT_HERSHEY_TRIPLEX,
                    0.5, (0, 0, 0), 1, cv2.LINE_AA)
        text_position += 20
    return frame


class EmotionVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.frame_count = 0
        self.last_result = None
        self.process_every_n_frames = 5  # skip frames to keep it responsive

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        self.frame_count += 1
        # Only run the (slow) DeepFace analysis every N frames
        if self.frame_count % self.process_every_n_frames == 0:
            try:
                result = analyze_frame(img)
                self.last_result = result
            except Exception as e:
                print(f"[DeepFace ERROR]: {e}")

        result = self.last_result
        if result:
            try:
                face_coordinates = result[0]["region"]
                x, y, w, h = (face_coordinates['x'], face_coordinates['y'],
                              face_coordinates['w'], face_coordinates['h'])
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

                text = f"{result[0]['dominant_emotion']}"
                cv2.putText(img, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 0), 1, cv2.LINE_AA)

                texts = [
                    f"Age: {result[0]['age']}",
                    f"Face Confidence: {round(result[0]['face_confidence'], 3)}",
                    f"Gender: {result[0]['dominant_gender']} "
                    f"{round(result[0]['gender'][result[0]['dominant_gender']], 3)}",
                    f"Race: {result[0]['dominant_race']}",
                    f"Dominant Emotion: {result[0]['dominant_emotion']} "
                    f"{round(result[0]['emotion'][result[0]['dominant_emotion']], 1)}",
                ]
                img = overlay_text_on_frame(img, texts)
            except (KeyError, IndexError):
                pass

        return av.VideoFrame.from_ndarray(img, format="bgr24")


def facesentiment():
    webrtc_streamer(
        key="emotion-detection",
        video_processor_factory=EmotionVideoProcessor,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
    )


def main():
    st.title("🎥 Real Time Emotion Detection")
    facesentiment()


if __name__ == "__main__":
    main()
