import streamlit as st
import cv2
import os
import base64
from openai import OpenAI


from dotenv import load_dotenv
load_dotenv()

# Initialize OpenAI client (make sure you set OPENAI_API_KEY in env)
client = OpenAI()

st.title("Football Video Analyzer Sample")

# Upload video
uploaded_file = st.file_uploader("Upload a football video", type=["mp4", "mov", "avi"])

def extract_frames(video_path, num_frames=3):
    """Extract evenly spaced frames from a video file."""
    frames = []
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # pick frame positions evenly spaced
    step = max(total_frames // (num_frames + 1), 1)
    for i in range(1, num_frames + 1):
        frame_no = i * step
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        success, frame = cap.read()
        if success:
            frame_path = f"frame_{i}.jpg"
            cv2.imwrite(frame_path, frame)
            frames.append(frame_path)
    cap.release()
    return frames

def analyze_frame_with_gpt(image_path):
    """Send one frame to GPT for analysis."""
    with open(image_path, "rb") as f:
        img_bytes = f.read()
    img_b64 = base64.b64encode(img_bytes).decode()

    response = client.chat.completions.create(
        model="gpt-4.1",  # GPT with vision
        messages=[
            {"role": "system", "content": "You are Football OS, an AI football coach. Always return structured analysis."},
            {"role": "user", "content": [
                {
                    "type": "text",
                    "text": (
                        "Analyze this football frame. "
                        "Provide output strictly in this format:\n\n"
                        "Detected action: ...\n"
                        "Outcome: ...\n"
                        "Strong point: ...\n"
                        "Weak point: ..."
                    )
                },
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
            ]}
        ]
    )
    return response.choices[0].message.content

if uploaded_file is not None:
    # Save uploaded file
    video_path = os.path.join("temp_video.mp4")
    with open(video_path, "wb") as f:
        f.write(uploaded_file.read())

    st.video(video_path)

    st.write("Extracting frames...")
    frames = extract_frames(video_path, num_frames=3)

    for frame in frames:
        st.image(frame, caption=f"Frame: {frame}")
        with st.spinner("Analyzing with Videos..."):
            feedback = analyze_frame_with_gpt(frame)
        st.write(" Feedback:")
        st.success(feedback)
