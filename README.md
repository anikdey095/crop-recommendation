# 🌾 CropPulse AI — Precision Crop Recommendation System

> **Production-Ready Machine Learning Platform & Vercel Serverless Microservice**  
> Evaluates soil macronutrients ($N, P, K$) and micro-climatic environmental dynamics to recommend optimal crops with **99.55% accuracy**.

[![Accuracy](https://img.shields.io/badge/Model_Accuracy-99.55%25-10B981?style=for-the-badge&logo=scikitlearn)](model_metrics.json)
[![Vercel Ready](https://img.shields.io/badge/Vercel-Serverless_Ready-black?style=for-the-badge&logo=vercel)](vercel.json)
[![Python](https://img.shields.io/badge/Python-3.9%20|%203.10%20|%203.11-3776AB?style=for-the-badge&logo=python)](requirements.txt)
[![Dual Engine](https://img.shields.io/badge/Inference_Engine-Dual_Engine_(Sklearn_+_Zero--Dep)-06B6D4?style=for-the-badge)](api/index.py)

---

## 🌟 Key Highlights

- **99.55% Test Accuracy / 99.59% 5-Fold Stratified Cross-Validation**: High-performance Random Forest ensemble trained on 2,200 agricultural samples across 22 crop classes.
- **Vercel Serverless Ready**: Native integration with `@vercel/python` and global Edge CDN static asset distribution.
- **Dual-Engine Inference Guarantee**:
  1. *Primary Engine*: Standard `scikit-learn` & `joblib` compressed model.
  2. *Serverless Fallback Engine*: Zero-dependency pure JSON decision tree evaluator (`0.68 ms` execution latency, 0MB C-extension overhead) ensuring 100% serverless uptime with zero cold-start timeouts.
- **Interactive Web Application**:
  - Live two-way synchronized nutrient sliders and number inputs.
  - 10 One-Click real-world agro-ecological presets (Monsoon Rice, Highland Coffee, Black Soil Cotton, Golden Jute, etc.).
  - Real-time meteorological autofill using Open-Meteo & GPS geolocation.
  - Parameter alignment and nutrient gap analysis with tailored fertilizer schedules (Urea, DAP, MOP, Lime).
  - Crop Encyclopedia with category filters and interactive benchmarks for all 22 crops.
  - Model & Data Analytics explorer with feature importance visualizations.
  - Dark / Light mode glassmorphism design with responsive layout.

---

## 🏗️ Project Architecture

```
crop-recommendation/
├── api/                        # Vercel Serverless Functions
│   ├── index.py                # WSGI / Flask serverless handler & API router
│   ├── model.joblib            # Compressed Scikit-Learn Random Forest model (374 KB)
│   ├── model_data.json         # Serialized decision tree ensemble for pure Python/JS inference
│   ├── crop_metadata.json      # Agronomic guidelines, optimal ranges & descriptions for 22 crops
│   └── model_metrics.json      # Accuracy benchmarks, feature importance & dataset ranges
├── public/                     # Static edge assets (also served by Vercel Edge CDN)
│   ├── index.html              # Modern, accessible semantic HTML5 single-page application
│   ├── style.css               # Vanilla CSS design system (glassmorphism tokens, dark/light theme)
│   └── app.js                  # Frontend state management, 2-way input sync, and API integration
├── index.html                  # Root static entry point for zero-config Vercel deployment
├── style.css                   # Root styles
├── app.js                      # Root logic
├── app.py                      # Local development server (http://127.0.0.1:5000)
├── train.py                    # Automated model training & asset serialization pipeline
├── test_app.py                 # Comprehensive 10-test automated verification suite
├── requirements.txt            # Pinned production Python dependencies
├── vercel.json                 # Vercel routing, rewrites, and serverless configuration
├── .vercelignore               # Excludes datasets, notebooks, and temporary files from build bundle
└── data.csv                    # 2,200 sample soil and climate training dataset
```

---

## 🚀 Quick Start (Local Development)

### 1. Clone & Set Up Environment
```bash
git clone https://github.com/anikdey095/crop-recommendation.git
cd crop-recommendation

# Create and activate virtual environment (optional but recommended)
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Application
```bash
python app.py
```
Open **[http://127.0.0.1:5000](http://127.0.0.1:5000)** in your browser.

### 3. Run Automated Tests
```bash
python test_app.py
```
*Executes all 10 unit and integration tests (model artifact validation, health checks, prediction accuracy, error handling, and sub-millisecond inference benchmarks).*

---

## ☁️ Deploying to Vercel

This repository is pre-configured for **zero-configuration deployment** to Vercel.

### Method A: Deploy via Vercel Web Dashboard (Recommended)

1. Push your code to your GitHub repository:
   ```bash
   git add .
   git commit -m "Production release ready for Vercel deployment"
   git push origin main
   ```
2. Navigate to [vercel.com/new](https://vercel.com/new).
3. Import your `crop-recommendation` repository.
4. Leave **Framework Preset** as **Other** (Vercel automatically detects `vercel.json`, `requirements.txt`, and `api/index.py`).
5. Click **Deploy**. Your application will be live globally in under a minute!

### Method B: Deploy via Vercel CLI

```bash
# Install Vercel CLI if not already installed
npm install -g vercel

# Log in and deploy
vercel
```

---

## 📊 Model Evaluation & Benchmarks

The model was evaluated against multiple baseline algorithms on a 20% stratified test split (440 samples) and 5-fold cross-validation:

| Model | Test Accuracy | 5-Fold Stratified CV | Inference Time | Deployment Role |
| :--- | :---: | :---: | :---: | :---: |
| **🌲 Random Forest (80 Trees)** | **99.55%** | **99.59% ± 0.3%** | **0.68 ms** | **Production (Active)** |
| **🌿 Decision Tree** | 97.95% | 98.77% ± 0.7% | 0.21 ms | Baseline Benchmark |
| **📈 Logistic Regression** | 95.00% | 97.36% ± 0.2% | 0.15 ms | Linear Baseline |

### Feature Importance Breakdown
1. **Rainfall**: `23.39%` (Critical for moisture threshold separation)
2. **Relative Humidity**: `22.55%` (Separates arid vs tropical crops)
3. **Potassium ($K$)**: `17.49%` (Strong marker for fruits such as Apple, Grapes, and Banana)
4. **Phosphorus ($P$)**: `15.04%` (Root development discriminator)
5. **Nitrogen ($N$)**: `9.50%` (Vegetative biomass indicator)
6. **Temperature**: `7.27%` (Thermal zone separator)
7. **Soil pH**: `4.76%` (Acidity/alkalinity tolerance)

---

## 📡 API Reference

### 1. Predict Crop
- **Endpoint**: `POST /api/predict`
- **Headers**: `Content-Type: application/json`
- **Request Body**:
```json
{
  "N": 90,
  "P": 42,
  "K": 43,
  "temperature": 24.5,
  "humidity": 82.0,
  "ph": 6.5,
  "rainfall": 202.9
}
```

- **Response (`200 OK`)**:
```json
{
  "success": true,
  "engine": "scikit-learn",
  "prediction": {
    "crop": "rice",
    "name": "Rice (Paddy)",
    "confidence": 91.67,
    "suitability_score": 96.0,
    "category": "Cereal / Grain",
    "season": "Kharif (Monsoon)",
    "growth_duration": "100 - 150 days",
    "water_requirement": "High (Standing water 5-10 cm during vegetative phase)",
    "soil_type": "Clayey, alluvial soil with high water retention",
    "fertilizer_guide": "High Nitrogen (Split application: basal, tillering, panicle), moderate P & K.",
    "economic_value": "Primary staple food crop; high domestic demand and stable market price.",
    "description": "Rice is the primary dietary staple for over half the world's population..."
  },
  "alternates": [
    { "crop": "jute", "name": "Jute (Golden Fiber)", "confidence": 8.33, "category": "Fiber / Cash Crop" }
  ],
  "suitability_analysis": {
    "N": { "status": "Optimal", "badge": "success", "user_value": 90.0, "advisory": "Current Nitrogen is in ideal range." },
    "P": { "status": "Optimal", "badge": "success", "user_value": 42.0, "advisory": "Current Phosphorus is in ideal range." },
    "K": { "status": "Optimal", "badge": "success", "user_value": 43.0, "advisory": "Current Potassium is in ideal range." }
  }
}
```

### 2. Available Crops Catalog
- **Endpoint**: `GET /api/crops`
- Returns metadata for all 22 supported crops.

### 3. Presets & Scenarios
- **Endpoint**: `GET /api/presets`
- Returns pre-configured realistic soil and weather presets.

### 4. Model Metadata & Metrics
- **Endpoint**: `GET /api/model-info`
- Returns model architecture, accuracy stats, and feature importance.

### 5. Health Check
- **Endpoint**: `GET /api/health`
- Returns `{ "status": "healthy", "model_loaded": true, "crops_count": 22 }`.

---

## 🌾 Supported Crops (22 Classes)

| Cereals & Grains | Pulses & Legumes | Fruits & Horticulture | Plantation & Cash |
| :--- | :--- | :--- | :--- |
| • Rice<br>• Maize | • Chickpea<br>• Kidney Beans<br>• Pigeon Peas<br>• Moth Beans<br>• Mung Bean<br>• Black Gram<br>• Lentil | • Pomegranate<br>• Banana<br>• Mango<br>• Grapes<br>• Watermelon<br>• Muskmelon<br>• Apple<br>• Orange<br>• Papaya | • Coconut<br>• Cotton<br>• Jute<br>• Coffee |

---

## 📄 License
This project is licensed under the MIT License.
