import streamlit as st
import tempfile

def main():
    st.set_page_config(page_title="Drone Footage Dashboard", layout="wide")
    st.sidebar.title("Dashboard Options")
    st.sidebar.markdown("""
    **Instructions:**
    - Upload an MP4 video file captured from your drone.
    - The video will be displayed below for review.
    
    **Future Features:**
    - Drone metadata (GPS, altitude, etc.)
    - Analytics and frame extraction
    """)
    st.markdown('<div style="text-align:center;font-size:2.5rem;font-weight:700;margin-bottom:0.5em;">Drone Footage Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div style="text-align:center;color:#555;margin-bottom:1.5em;">Upload and view your drone-captured MP4 video footage below.</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose a drone-captured MP4 video file", type=['mp4'])
    if uploaded_file is not None:
        st.markdown(
            """
            <style>
            .centered-video-container {
                display: flex;
                justify-content: center;
                align-items: center;
                width: 100%;
            }
            .centered-video-container video {
                width: 640px !important;
                height: 360px !important;
                max-width: 100%;
            }
            </style>
            <div class="centered-video-container">
            """,
            unsafe_allow_html=True,
        )
        st.video(uploaded_file)
        st.markdown("</div>", unsafe_allow_html=True)
        st.success("Video loaded successfully! Review your footage above.")
    else:
        st.info("Please upload a drone-captured MP4 video file to begin.")

if __name__ == "__main__":
    main()