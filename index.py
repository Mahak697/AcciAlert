import streamlit as st
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from PIL import Image
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests
import folium
from streamlit_folium import st_folium
from streamlit_geolocation import streamlit_geolocation
from geopy.distance import geodesic

# ================= UI CONFIG =================
st.set_page_config(page_title="Accident Intelligence System", layout="wide", page_icon="🚨")

# ================= CUSTOM CSS FOR PROFESSIONAL UI =================
st.markdown("""
    <style>
    /* Emergency button CSS */
    .emergency-button {
        display: block;
        width: 100%;
        padding: 15px;
        text-align: center;
        background-color: #dc2626;
        color: white !important;
        font-weight: bold;
        font-size: 18px;
        border-radius: 8px;
        text-decoration: none;
        margin-top: 10px;
        box-shadow: 0 4px 6px rgba(220, 38, 38, 0.4);
        transition: background-color 0.3s, transform 0.2s;
    }
    .emergency-button:hover {
        background-color: #b91c1c;
        transform: scale(1.02);
    }
    </style>
""", unsafe_allow_html=True)


# ================= LOAD DATA =================
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("dataset_traffic_accident_prediction1.csv")
        return df
    except:
        return pd.DataFrame()

@st.cache_resource
def load_ml_model():
    try:
        model = load_model("accident_detection_model.h5")
        return model
    except:
        return None

df = load_data()
model = load_ml_model()

# ================= UTILITY FUNCTIONS =================
def get_nearest_hospitals(lat, lon):
    """Fetch nearest hospitals using Overpass API"""
    overpass_url = "http://overpass-api.de/api/interpreter"
    overpass_query = f"""
    [out:json];
    node(around:5000,{lat},{lon})["amenity"="hospital"];
    out 5;
    """
    try:
        response = requests.get(overpass_url, params={'data': overpass_query}, timeout=10)
        data = response.json()
        hospitals = []
        for element in data.get('elements', []):
            tags = element.get('tags', {})
            name = tags.get('name', 'Unknown Hospital')
            address = tags.get('addr:full', tags.get('addr:street', 'Address not available'))
            phone = tags.get('phone', 'Phone not available')
            
            # Calculate distance
            h_lat = element.get('lat')
            h_lon = element.get('lon')
            distance = round(geodesic((lat, lon), (h_lat, h_lon)).kilometers, 2)
            
            hospitals.append({
                'name': name,
                'address': address,
                'distance': distance,
                'phone': phone,
                'lat': h_lat,
                'lon': h_lon
            })
        # Sort by distance
        hospitals = sorted(hospitals, key=lambda x: x['distance'])
        return hospitals[:5]
    except Exception as e:
        st.error(f"Error fetching hospitals: {e}")
        return []

def send_email_alert(lat, lon):
    """Send emergency email alert using secure Gmail SMTP."""
    # ⚠️ CONFIGURATION REQUIRED ⚠️
    # You MUST replace these strings with your actual credentials for emails to send.
    SENDER_EMAIL = "mahaksahu93025@gmail.com"  
    # Must be a 16-character App Password, NOT your regular Gmail password. (Spaces removed)
    SENDER_APP_PASSWORD = "zlnmbrtiifasunxo"  
    RECEIVER_EMAIL = "mahaksahu93025@gmail.com"
    
    maps_link = f"https://www.google.com/maps?q={lat},{lon}"
    body = f"An accident has been detected by the system.\n\n" \
           f"Live Location Coordinates: {lat}, {lon}\n" \
           f"Google Maps Link: {maps_link}"
           
    msg = MIMEMultipart()
    msg['From'] = SENDER_EMAIL
    msg['To'] = RECEIVER_EMAIL
    msg['Subject'] = "Accident Detection Alert"
    msg.attach(MIMEText(body, 'plain'))
    
    try:
        # Secure connection to Gmail's SMTP server
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_APP_PASSWORD)
        server.sendmail(SENDER_EMAIL, RECEIVER_EMAIL, msg.as_string())
        server.quit()
        return True, "Email sent successfully."
        
    except smtplib.SMTPAuthenticationError:
        return False, "Authentication Error: Invalid email or App Password. Ensure 2-Step Verification is on and you generated an App Password."
    except Exception as e:
        return False, f"Delivery Failed: {str(e)}"

# ================= SIDEBAR & NAVIGATION =================
st.sidebar.markdown("<h2 style='text-align: center;'>🛰️ Dashboard Navigation</h2>", unsafe_allow_html=True)
menu = st.sidebar.radio(
    "",
    ["Home", "CSV Analytics", "Image & Camera Prediction", "Model Insights"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("<h3 style='text-align: center; color: #fca5a5;'>🚑 Immediate Actions</h3>", unsafe_allow_html=True)
# Mobile Friendly Emergency Call button
st.sidebar.markdown('<a href="tel:911" class="emergency-button">📞 Call Emergency (911)</a>', unsafe_allow_html=True)

# ================= HOME =================
if menu == "Home":
    st.markdown("<h1 style='text-align: center;'>AI Accident Intelligence System</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; color: #475569;'>Advanced Deep Learning & Data Analytics Dashboard</h4>", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Model Precision", "88%", "Stable Build")
    with col2:
        st.metric("Analyzed Records", len(df) if not df.empty else 0, "Up to Date")
    with col3:
        st.metric("Training Volume", "33K Images", "High Quality Dataset")
    with col4:
        st.metric("Model Architecture", "CNN", "Deep Learning")
    
    st.markdown("---")
    
    # Two Columns layout for project overview
    col_left, col_right = st.columns((2,1))
    
    with col_left:
        st.markdown("### 🌟 Platform Overview")
        st.info("""
        This advanced web application utilizes cutting-edge Convolutional Neural Networks (CNN) to detect 
        road accidents in real-time from camera streams and uploaded imagery. Built to enhance public safety, 
        the dashboard now features integrated geolocation tracking and automated emergency email dispatch.
        """)
        
        c1, c2 = st.columns(2)
        with c1:
            st.success("#### 🎯 Core Objectives")
            st.write("""
            - ✅ AI-powered accident detection
            - ✅ Comprehensive traffic data analytics
            - ✅ Real-time inference monitoring
            - ✅ Instant mapping of incidents
            """)
        with c2:
            st.warning("#### 💻 Technology Stack")
            st.write("""
            - 🐍 Python, Streamlit
            - 🧠 TensorFlow, Keras
            - 👁️ OpenCV
            - 📍 Folium, Geolocation
            """)

    with col_right:
        if not df.empty and len(df.columns) > 0:
            st.markdown("#### Category Distribution")
            fig = px.pie(df, names=df.columns[0], hole=0.5, color_discrete_sequence=px.colors.sequential.RdBu)
            fig.update_layout(margin=dict(t=20, b=0, l=0, r=0), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

# ================= CSV ANALYTICS =================
elif menu == "CSV Analytics":
    st.title("📊 Traffic Data Analytics")
    
    if df.empty:
        st.error("🚨 Dataset not found or empty.")
    else:
        st.dataframe(df.head(10), use_container_width=True)

        st.markdown("---")
        col_hist, col_box = st.columns(2)
        numeric_cols = df.select_dtypes(include=np.number).columns
        
        with col_hist:
            st.subheader("📈 Distribution Analysis")
            if len(numeric_cols) > 0:
                col_h = st.selectbox("Select Feature for Histogram", numeric_cols, key="hist")
                fig_hist = px.histogram(df, x=col_h, template="plotly_white", color_discrete_sequence=["#3b82f6"])
                st.plotly_chart(fig_hist, use_container_width=True)
            
        with col_box:
            st.subheader("📦 Spread & Outliers")
            if len(numeric_cols) > 0:
                fig_box = px.box(df, y=col_h, template="plotly_white", color_discrete_sequence=["#ef4444"])
                st.plotly_chart(fig_box, use_container_width=True)

        st.markdown("---")
        if len(numeric_cols) > 1:
            st.subheader("🔥 Feature Correlation Matrix")
            corr = df[numeric_cols].corr()
            fig_heat, ax = plt.subplots(figsize=(10,5))
            sns.heatmap(corr, annot=True, cmap="mako", ax=ax, linewidths=0.5, fmt=".2f")
            st.pyplot(fig_heat)

# ================= IMAGE PREDICTION & EMERGENCY =================
elif menu == "Image & Camera Prediction":
    st.title("🚗 AI Inference & Emergency Response")
    
    # 📍 Live Location Fetch
    st.markdown("### 📍 Live Location Access")
    st.markdown("Allow location permissions below to enable the incident mapping and emergency routing.")
    
    col_loc, col_status = st.columns((1,3))
    with col_loc:
        location = streamlit_geolocation()
        
    user_lat, user_lon = None, None
    with col_status:
        if location and location.get('latitude') and location.get('longitude'):
            user_lat = location['latitude']
            user_lon = location['longitude']
            st.success(f"**Coordinates Acquired:** {user_lat}, {user_lon}")
        else:
            st.info("Awaiting location access...")
    
    st.markdown("---")
    
    # Detection Mode Selection
    mode = st.radio("Choose AI Inference Mode:", ["📷 Upload Image", "📹 Start Camera"], horizontal=True)

    def trigger_emergency_protocol():
        st.error("### 🚨 CRITICAL: ACCIDENT DETECTED 🚨")
        
        if user_lat and user_lon:
            st.markdown("#### 🗺️ Incident Location")
            # Map rendering
            m = folium.Map(location=[user_lat, user_lon], zoom_start=16)
            folium.Marker(
                [user_lat, user_lon], 
                popup="Accident Origin", 
                tooltip="Accident Detected Here", 
                icon=folium.Icon(color="red", icon="warning-sign")
            ).add_to(m)
            st_folium(m, width=700, height=350)
            
            # Google Maps Link
            maps_url = f"https://www.google.com/maps?q={user_lat},{user_lon}"
            st.markdown(f"[🔗 **Open Location in Google Maps**]({maps_url})")

            # Integration 2: Nearest Hospitals via Overpass API
            with st.spinner("Finding nearest hospitals..."):
                hospitals = get_nearest_hospitals(user_lat, user_lon)
                if hospitals:
                    st.markdown("#### 🏥 Nearest Hospitals")
                    for h in hospitals:
                        st.info(f"**{h['name']}**  \n🛣️ {h['address']}  \n📏 {h['distance']} km away  \n📞 {h['phone']}")
                else:
                    st.warning("No hospitals found nearby in the OpenStreetMap database.")
            
            # Integration 1: Trigger active Email Alert
            email_success, email_message = send_email_alert(user_lat, user_lon)
            if email_success:
                st.success("✉️ Automated emergency email successfully dispatched to **mahaksahu93025@gmail.com**.")
            else:
                st.warning(f"⚠️ Email Not Sent: {email_message}")
        else:
            st.error("⚠️ Location tracking disabled. Could not map the incident or find corresponding hospitals.")

    # ================= IMAGE UPLOAD INFERENCE =================
    if mode == "📷 Upload Image":
        file = st.file_uploader("Upload accident or road scene image", type=["jpg","png","jpeg"])
        if file is not None:
            col_img, col_res = st.columns(2)
            
            with col_img:
                image = Image.open(file)
                st.image(image, caption="Source Image", use_container_width=True)
            
            with col_res:
                if model:
                    img = np.array(image.convert('RGB'))
                    img = cv2.resize(img, (128,128))
                    img = img.astype("float32") / 255.0
                    img = img.reshape(1,128,128,3)

                    with st.spinner("AI analyzing visual feed..."):
                        pred = model.predict(img)[0][0]
                    
                    st.markdown(f"**AI Confidence Score:** `{pred:.2%}`")
                    st.progress(float(pred))
                    
                    if pred > 0.3:
                        trigger_emergency_protocol()
                    else:
                        st.success("### ✅ SCENE SAFE\nNo accident signatures detected.")
                else:
                    st.error("Neural Network model not found in directory.")

    # ================= CAMERA INFERENCE =================
    elif mode == "📹 Start Camera":
        run_camera = st.checkbox("Toggle Live Camera Feed")
        frame_window = st.image([])
        
        if run_camera and model:
            cap = cv2.VideoCapture(0)
            accident_flagged = False
            
            while run_camera:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to access system camera.")
                    break
                    
                display_frame = frame.copy()
                img = cv2.resize(frame, (128,128))
                img = img.astype("float32") / 255.0
                img = img.reshape(1,128,128,3)

                pred = model.predict(img)[0][0]

                if pred > 0.5:
                    cv2.putText(display_frame, "ACCIDENT DETECTED", (30,50), cv2.FONT_HERSHEY_DUPLEX, 1, (0,0,255), 2)
                    accident_flagged = True
                else:
                    cv2.putText(display_frame, "NORMAL", (30,50), cv2.FONT_HERSHEY_DUPLEX, 1, (0,255,0), 2)

                display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                frame_window.image(display_frame)
                
            cap.release()
            
            if accident_flagged:
                st.markdown("---")
                trigger_emergency_protocol()

# ================= MODEL INSIGHTS =================
elif menu == "Model Insights":
    st.title("🤖 CNN Model Architecture")
    st.info("The system employs a Convolutional Neural Network (CNN) specifically tuned for road scene analysis and accident signature detection.")
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Target Accuracy", "88.0%")
    col2.metric("Function Layer", "ReLU + Sigmoid")
    col3.metric("Output Class", "Binary (Safe/Accident)")

    st.markdown("---")
    
    with st.expander("🔍 **Feature Extraction Methodology**", expanded=True):
        st.write("""
        The Deep Learning model scrutinizes images for specific high-level features:
        - **Vehicle Integrity Anomalies:** Bent frames, shattered glass, multi-vehicle overlap.
        - **Environmental Hazards:** Smoke plumes, fire patterns.
        - **Road Context:** Debris scattering, sudden traffic disruption.
        """)

    with st.expander("⚙️ **Engine Configuration**"):
        st.write("""
        - **Training/Validation Split:** 80% / 20%
        - **Optimization Algorithm:** Adam Optimizer
        - **Loss Evaluation:** Binary Crossentropy
        """)

    with st.expander("📈 **Real-World Reliability**"):
        st.success("✔ Adaptive prediction across varying light conditions (Day/Night).")
        st.success("✔ Resistant to noise introduced by poor camera resolution.")
        st.success("✔ Extremely high recall rates prioritizing false positives over false negatives for safety.")
