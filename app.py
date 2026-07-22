import av
import cv2
import streamlit as st
from deepface import DeepFace
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase


def overlay_text_on_frame(frame, texts):
    """Draw a translucent header bar with a few lines of text on top of the frame."""
    overlay = frame.copy()
    alpha = 0.9
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 100), (255, 255, 255), -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    text_position = 15
    for text in texts:
        cv2.putText(
            frame, text, (10, text_position),
            cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA
        )
        text_position += 20
    return frame


class EmotionProcessor(VideoProcessorBase):
    """
    Runs once per incoming video frame from the browser.
    All DeepFace calls are wrapped so a single bad/faceless frame
    never crashes the whole stream.
    """

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        try:
            result = DeepFace.analyze(
                img_path=img,
                actions=['age', 'gender', 'race', 'emotion'],
                enforce_detection=False,
                detector_backend="opencv",
                align=True,
                silent=True,
            )
        except Exception:
            # Analysis failed for this frame (e.g. decoder hiccup) -- skip it, keep streaming
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        # DeepFace can return an empty list, or a region with ~0 confidence,
        # when no face is actually present -- guard against both.
        if not result:
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        face = result[0]
        region = face.get("region", {})
        w, h = region.get("w", 0), region.get("h", 0)

        # Skip drawing/overlay if there's effectively no real face box
        if w <= 0 or h <= 0:
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        x, y = region.get("x", 0), region.get("y", 0)

        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            img, face["dominant_emotion"], (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA
        )

        texts = [
            f"Age: {face['age']}",
            f"Face Confidence: {round(face.get('face_confidence', 0), 3)}",
            f"Gender: {face['dominant_gender']} "
            f"{round(face['gender'][face['dominant_gender']], 3)}",
            f"Race: {face['dominant_race']}",
            f"Dominant Emotion: {face['dominant_emotion']} "
            f"{round(face['emotion'][face['dominant_emotion']], 1)}",
        ]
        img = overlay_text_on_frame(img, texts)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


def main():
    st.title("🎥 Real Time Emotion Detection")
    st.caption(
        "Uses your browser's camera (via WebRTC) so this also works "
        "when the app is deployed to Streamlit Cloud."
    )
    webrtc_streamer(
        key="emotion-detection",
        video_processor_factory=EmotionProcessor,
        media_stream_constraints={"video": True, "audio": False},
    )


if __name__ == "__main__":
    main()
