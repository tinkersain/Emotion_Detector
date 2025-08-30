import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
from collections import deque
import time
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Real-time Emotion Detection",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for eye-catching UI
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF6B6B;
        text-align: center;
        font-weight: bold;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .emotion-box {
        background: linear-gradient(45deg, #FE6B8B 30%, #FF8E53 90%);
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
        color: white;
        font-size: 1.5rem;
        font-weight: bold;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
    }
    
    .stats-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        border-left: 5px solid #4CAF50;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #2196F3 0%, #21CBF3 100%);
    }
</style>
""", unsafe_allow_html=True)

class EmotionDetector:
    def __init__(self, model_path):
        """Initialize the emotion detector with the trained model."""
        self.model = tf.keras.models.load_model(model_path)
        # Common emotion labels - adjust based on your model
        self.emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Initialize metrics tracking
        self.emotion_history = deque(maxlen=50)
        self.confidence_history = deque(maxlen=50)
        self.detection_times = deque(maxlen=50)
        
    def preprocess_image(self, image):
        """Preprocess image for model prediction."""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Resize to model input size (adjust based on your model)
        resized = cv2.resize(gray, (48, 48))  # Common size for emotion detection
        normalized = resized / 255.0
        return np.expand_dims(np.expand_dims(normalized, axis=0), axis=-1)
    
    def detect_faces(self, image):
        """Detect faces in the image."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        return faces
    
    def predict_emotion(self, face_image):
        """Predict emotion from face image."""
        start_time = time.time()
        
        preprocessed = self.preprocess_image(face_image)
        predictions = self.model.predict(preprocessed, verbose=0)
        
        detection_time = time.time() - start_time
        
        # Get prediction results
        emotion_idx = np.argmax(predictions[0])
        confidence = np.max(predictions[0])
        emotion = self.emotion_labels[emotion_idx]
        
        # Store metrics
        self.emotion_history.append(emotion)
        self.confidence_history.append(confidence)
        self.detection_times.append(detection_time)
        
        return emotion, confidence, predictions[0], detection_time

def create_emotion_chart(predictions, labels):
    """Create a bar chart for emotion predictions."""
    fig = go.Figure(data=[
        go.Bar(
            x=labels,
            y=predictions,
            marker_color=px.colors.qualitative.Set3[:len(labels)],
            text=[f'{p:.2f}' for p in predictions],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Emotion Prediction Confidence",
        xaxis_title="Emotions",
        yaxis_title="Confidence",
        showlegend=False,
        height=400,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    
    return fig

def create_history_chart(history, title):
    """Create a line chart for historical data."""
    if len(history) < 2:
        return None
        
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=list(history),
        mode='lines+markers',
        name=title,
        line=dict(color='#FF6B6B', width=3),
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        title=f"{title} Over Time",
        xaxis_title="Frame",
        yaxis_title=title,
        height=300,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    
    return fig

def main():
    # Header
    st.markdown('<h1 class="main-header">🎭 Real-time Emotion Detection</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.markdown("## 🔧 Configuration")
    
    # Model loading
    if 'detector' not in st.session_state:
        model_path = st.sidebar.text_input("Model Path", value="model/emotion_model.h5")
        
        if st.sidebar.button("Load Model"):
            try:
                with st.spinner("Loading model..."):
                    st.session_state.detector = EmotionDetector(model_path)
                st.sidebar.success("Model loaded successfully!")
            except Exception as e:
                st.sidebar.error(f"Error loading model: {str(e)}")
                return
    
    if 'detector' not in st.session_state:
        st.warning("Please load your emotion detection model first!")
        return
    
    detector = st.session_state.detector
    
    # Sidebar controls
    confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5)
    show_metrics = st.sidebar.checkbox("Show Advanced Metrics", True)
    show_history = st.sidebar.checkbox("Show Historical Data", True)
    
    # Main layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📹 Live Camera Feed")
        
        # Camera input
        run_camera = st.checkbox("Start Camera")
        
        if run_camera:
            # Use camera
            camera_input = st.camera_input("Take a picture")
            
            if camera_input is not None:
                # Convert to OpenCV format
                image = Image.open(camera_input)
                image_array = np.array(image)
                
                # Detect faces
                faces = detector.detect_faces(image_array)
                
                if len(faces) > 0:
                    # Process the largest face
                    largest_face = max(faces, key=lambda x: x[2] * x[3])
                    x, y, w, h = largest_face
                    
                    # Extract face region
                    face_img = image_array[y:y+h, x:x+w]
                    
                    # Predict emotion
                    emotion, confidence, predictions, detection_time = detector.predict_emotion(face_img)
                    
                    # Draw bounding box and label
                    cv2.rectangle(image_array, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(image_array, f'{emotion} ({confidence:.2f})', 
                              (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    
                    # Display image
                    st.image(image_array, channels="RGB", use_column_width=True)
                    
                    # Current emotion display
                    if confidence >= confidence_threshold:
                        st.markdown(f'<div class="emotion-box">Current Emotion: {emotion}</div>', 
                                  unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="emotion-box">Low Confidence Detection</div>', 
                                  unsafe_allow_html=True)
                    
                else:
                    st.image(image_array, channels="RGB", use_column_width=True)
                    st.warning("No face detected in the image!")
        
        else:
            st.info("Enable camera to start emotion detection")
    
    with col2:
        st.markdown("### 📊 Live Analytics")
        
        if run_camera and camera_input is not None and len(faces) > 0:
            # Current prediction chart
            chart = create_emotion_chart(predictions, detector.emotion_labels)
            st.plotly_chart(chart, use_container_width=True)
            
            # Metrics
            if show_metrics:
                st.markdown("### 📈 Performance Metrics")
                
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown(f'<div class="metric-container">Confidence<br><b>{confidence:.2%}</b></div>', 
                              unsafe_allow_html=True)
                with col_b:
                    st.markdown(f'<div class="metric-container">Detection Time<br><b>{detection_time:.3f}s</b></div>', 
                              unsafe_allow_html=True)
                
                # Additional metrics
                if len(detector.emotion_history) > 0:
                    avg_confidence = np.mean(detector.confidence_history)
                    avg_detection_time = np.mean(detector.detection_times)
                    
                    st.markdown(f"**Average Confidence:** {avg_confidence:.2%}")
                    st.markdown(f"**Average Detection Time:** {avg_detection_time:.3f}s")
                    st.markdown(f"**Frames Processed:** {len(detector.emotion_history)}")
            
            # Historical data
            if show_history and len(detector.confidence_history) > 1:
                st.markdown("### 📈 Historical Trends")
                
                # Confidence history
                conf_chart = create_history_chart(detector.confidence_history, "Confidence")
                if conf_chart:
                    st.plotly_chart(conf_chart, use_container_width=True)
                
                # Emotion distribution
                if len(detector.emotion_history) > 0:
                    emotion_counts = pd.Series(detector.emotion_history).value_counts()
                    
                    pie_fig = px.pie(
                        values=emotion_counts.values,
                        names=emotion_counts.index,
                        title="Emotion Distribution"
                    )
                    pie_fig.update_layout(height=300)
                    st.plotly_chart(pie_fig, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("### 💡 Tips for Better Detection")
    tips_col1, tips_col2, tips_col3 = st.columns(3)
    
    with tips_col1:
        st.markdown("""
        **Lighting**
        - Ensure good lighting
        - Avoid harsh shadows
        - Face the light source
        """)
    
    with tips_col2:
        st.markdown("""
        **Positioning**
        - Keep face centered
        - Maintain steady position
        - Avoid extreme angles
        """)
    
    with tips_col3:
        st.markdown("""
        **Expression**
        - Express clearly
        - Hold expression briefly
        - Avoid rapid changes
        """)

if __name__ == "__main__":
    main()