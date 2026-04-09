# Full-Stack Accident Intelligence System Architecture

## System Architecture

The modernized application will utilize a decoupled Client-Server architecture:

1. **Frontend (React.js):** 
   - A modern Single Page Application (SPA) built with Vite and React.
   - Tailored UI for accessing the device camera and uploading photos.
   - Renders live maps (via `react-leaflet` or Google Maps API) and emergency response data.
   - Communicates with the backend exclusively via REST APIs.

2. **Backend (FastAPI):**
   - High-performance asynchronous Python web framework.
   - Rapid execution of CNN image inference using TensorFlow/Keras.
   - Handles external cross-origin requests (CORS) from the frontend.
   - Manages integrations: Overpass API for nearest hospitals, SMTP/Resend for email dispatch.

3. **Data Flow:**
   User -> Captures Image (React) -> Base64/Multipart Upload to API (`/predict-accident`) -> API infers probability (FastAPI + TF) -> If accident, API triggers Email & fetches Hospitals -> API returns JSON Payload -> React updates UI map and list.

---

## Project Folder Structure

```text
accident-intelligence-system/
│
├── backend/                       # Python FastAPI Backend
│   ├── main.py                    # Entry point, API routes
│   ├── requirements.txt           # Python dependencies
│   ├── models/
│   │   └── accident_detection_model.h5  # Existing CNN Model
│   ├── services/
│   │   ├── inference.py           # ML prediction logic
│   │   ├── location.py            # Overpass API logic (hospitals)
│   │   └── notification.py        # SMTP email dispatch logic
│   └── utils/                     # Helpers (image resizing, base64)
│
├── frontend/                      # React.js Frontend
│   ├── package.json
│   ├── index.html
│   ├── vite.config.js
│   ├── src/
│   │   ├── App.jsx                # Main Layout & Router
│   │   ├── main.jsx
│   │   ├── components/
│   │   │   ├── ImageUpload.jsx    # Image selection component
│   │   │   ├── CameraFeed.jsx     # Webcam component
│   │   │   ├── IncidentMap.jsx    # Map rendering
│   │   │   └── HospitalList.jsx   # Hospital display cards
│   │   └── styles/
│   │       └── index.css          # Tailwind or custom CSS
│
└── README.md                      # Instructions to run the stack
```

---

## Example Backend API Code (`backend/main.py`)

```python
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import cv2
import numpy as np
from tensorflow.keras.models import load_model

app = FastAPI(title="Accident Detection API")

# Enable CORS for React Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Change to frontend URL in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load existing model once at startup
model = load_model("models/accident_detection_model.h5")

class PredictionResponse(BaseModel):
    is_accident: bool
    confidence: float
    hospitals: list = []
    email_sent: bool = False

@app.post("/predict-accident", response_model=PredictionResponse)
async def predict_accident(
    file: UploadFile = File(...), 
    lat: float = Form(None), 
    lon: float = Form(None)
):
    # 1. Read Image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # 2. Preprocess
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (128, 128))
    img = img.astype("float32") / 255.0
    img = img.reshape(1, 128, 128, 3)

    # 3. Predict
    pred = model.predict(img)[0][0]
    is_accident = bool(pred > 0.3)

    response = PredictionResponse(is_accident=is_accident, confidence=float(pred))

    # 4. Trigger Emergency Protocol if applicable
    if is_accident and lat and lon:
        # Example pseudo-functions from your services directory
        # response.hospitals = get_nearest_hospitals(lat, lon)
        # response.email_sent = send_email_alert(lat, lon)
        pass

    return response
```

---

## Example Frontend Code (`frontend/src/App.jsx`)

```jsx
import React, { useState, useEffect } from 'react';
import axios from 'axios';

function App() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [location, setLocation] = useState({ lat: null, lon: null });
  const [loading, setLoading] = useState(false);

  // Auto-fetch location on load
  useEffect(() => {
    if ("geolocation" in navigator) {
      navigator.geolocation.getCurrentPosition((pos) => {
        setLocation({ lat: pos.coords.latitude, lon: pos.coords.longitude });
      });
    }
  }, []);

  const handleDiagnose = async () => {
    if (!file) return;
    setLoading(true);
    
    // Create multipart form payload
    const formData = new FormData();
    formData.append("file", file);
    if (location.lat) formData.append("lat", location.lat);
    if (location.lon) formData.append("lon", location.lon);

    try {
      const resp = await axios.post("http://localhost:8000/predict-accident", formData);
      setResult(resp.data);
    } catch (error) {
      console.error("API Error", error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-50 p-8 flex flex-col items-center">
      <h1 className="text-3xl font-bold mb-4">AI Accident Intelligence</h1>
      
      {/* Upload Section */}
      <div className="bg-white p-6 shadow-md rounded-lg w-full max-w-md">
         <input type="file" onChange={(e) => setFile(e.target.files[0])} className="mb-4 w-full"/>
         <button 
           onClick={handleDiagnose} 
           className="bg-blue-600 text-white w-full py-2 rounded font-bold hover:bg-blue-700 disabled:opacity-50"
           disabled={loading}
         >
           {loading ? "Analyzing Scene..." : "Diagnose Image"}
         </button>
      </div>

      {/* Results Section */}
      {result && (
        <div className="mt-8 w-full max-w-2xl">
           {result.is_accident ? (
             <div className="bg-red-100 border-l-4 border-red-500 text-red-700 p-4 mb-4">
               <h2 className="font-bold text-xl">🚨 CRITICAL: ACCIDENT DETECTED</h2>
               <p>Confidence: {(result.confidence * 100).toFixed(1)}%</p>
               {/* Insert Hospital List Component Here */}
               {/* Insert Map Component Here */}
               <a href="tel:911" className="block text-center mt-4 bg-red-600 text-white font-bold py-3 rounded">📞 Call 911 Now</a>
             </div>
           ) : (
             <div className="bg-green-100 border-l-4 border-green-500 text-green-700 p-4">
               <h2 className="font-bold text-xl">✅ SCENE SAFE</h2>
             </div>
           )}
        </div>
      )}
    </div>
  );
}

export default App;
```

---

## Instructions to Run the Project Locally

**1. Clone/Setup the Project:**
```bash
mkdir accident-intelligence-system
cd accident-intelligence-system
```

**2. Run the Backend:**
```bash
cd backend
python -m venv venv
venv\Scripts\activate      # On Windows
pip install fastapi uvicorn tensorflow opencv-python pydantic python-multipart
uvicorn main:app --reload
```
*(Backend runs on `http://localhost:8000`)*

**3. Run the Frontend:**
```bash
cd ../frontend
npm install react react-dom axios
npm install -D tailwindcss postcss autoprefixer vite
npx tailwindcss init -p
npm run dev
```
*(Frontend runs on `http://localhost:5173`)*
