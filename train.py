"""
Production Model Training Pipeline for Crop Recommendation
Trains Random Forest model with stratified k-fold validation, evaluates against Decision Tree and Logistic Regression,
and exports compressed joblib artifact, pure-JSON inference tree, and rich crop metadata.
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
import joblib

DATA_PATH = os.path.join(os.path.dirname(__file__), 'data.csv')
MODEL_JOBLIB_PATH = os.path.join(os.path.dirname(__file__), 'model.joblib')
MODEL_JSON_PATH = os.path.join(os.path.dirname(__file__), 'model_data.json')
METADATA_PATH = os.path.join(os.path.dirname(__file__), 'crop_metadata.json')
METRICS_PATH = os.path.join(os.path.dirname(__file__), 'model_metrics.json')

# Rich agronomic database for the 22 crops in data.csv
CROP_DETAILS = {
    "rice": {
        "common_name": "Rice (Paddy)",
        "category": "Cereal / Grain",
        "season": "Kharif (Monsoon)",
        "soil_type": "Clayey, alluvial soil with high water retention",
        "growth_duration": "100 - 150 days",
        "water_requirement": "High (Standing water 5-10 cm during vegetative phase)",
        "fertilizer_guide": "High Nitrogen (Split application: basal, tillering, panicle), moderate P & K.",
        "economic_value": "Primary staple food crop; high domestic demand and stable market price.",
        "description": "Rice is the primary dietary staple for over half the world's population. It thrives in high humidity, prolonged rainfall, and warm temperatures."
    },
    "maize": {
        "common_name": "Maize (Corn)",
        "category": "Cereal / Grain",
        "season": "Kharif / Rabi",
        "soil_type": "Well-drained deep loamy soil rich in organic matter",
        "growth_duration": "90 - 120 days",
        "water_requirement": "Moderate (500-800 mm evenly distributed)",
        "fertilizer_guide": "Balanced NPK (120:60:40 kg/ha); sensitive to zinc deficiency.",
        "economic_value": "High demand for food, livestock feed, biofuel, and starch industries.",
        "description": "Maize is a versatile cereal crop requiring warm sunshine and well-aerated soil. Cannot tolerate waterlogging."
    },
    "chickpea": {
        "common_name": "Chickpea (Gram / Chana)",
        "category": "Pulse / Legume",
        "season": "Rabi (Winter)",
        "soil_type": "Deep loamy or clay loam with good drainage (pH 6.0 - 8.0)",
        "growth_duration": "90 - 110 days",
        "water_requirement": "Low (Drought-tolerant, 2-3 irrigations sufficient)",
        "fertilizer_guide": "Low Nitrogen (fixes atmospheric N), high Phosphorus (40-60 kg P2O5/ha).",
        "economic_value": "Premium protein source with excellent export and domestic wholesale value.",
        "description": "Chickpea is a major winter pulse crop. Natural nitrogen-fixing ability enriches soil fertility for rotation crops."
    },
    "kidneybeans": {
        "common_name": "Kidney Beans (Rajma)",
        "category": "Pulse / Legume",
        "season": "Kharif / Rabi",
        "soil_type": "Light, loose, well-aerated sandy loam with pH 5.5 - 6.5",
        "growth_duration": "110 - 130 days",
        "water_requirement": "Moderate (Requires consistent moisture without waterlogging)",
        "fertilizer_guide": "Requires higher initial Nitrogen as nodulation is relatively poor compared to other pulses.",
        "economic_value": "High market retail value; popular delicacy with long storage life.",
        "description": "Kidney beans require cool to moderate climate and rich, well-draining soil. Highly sensitive to frost and waterlogging."
    },
    "pigeonpeas": {
        "common_name": "Pigeon Pea (Arhar / Toor Dal)",
        "category": "Pulse / Legume",
        "season": "Kharif",
        "soil_type": "Deep well-drained loam or alluvial soil (pH 6.5 - 7.5)",
        "growth_duration": "150 - 240 days",
        "water_requirement": "Moderate to Low (Deep tap root allows deep moisture mining)",
        "fertilizer_guide": "20-25 kg N, 50 kg P2O5 per hectare; Rhizobium seed inoculation recommended.",
        "economic_value": "Core daily protein pulse across South Asia; commanding resilient pricing.",
        "description": "Pigeon pea is an essential perennial-grown-as-annual pulse. Deep roots aerate soil and break up subsoil compaction."
    },
    "mothbeans": {
        "common_name": "Moth Bean (Matki)",
        "category": "Pulse / Legume",
        "season": "Kharif (Arid / Semi-arid)",
        "soil_type": "Sandy, sandy-loam, even low fertility soils",
        "growth_duration": "75 - 90 days",
        "water_requirement": "Very Low (One of the most drought-hardy legumes in existence)",
        "fertilizer_guide": "Minimal fertilizer needed; 10 kg N, 20 kg P2O5 per hectare.",
        "economic_value": "Valuable drought-resilience food and nutritious dryland fodder.",
        "description": "Extremely drought-resistant crop suited for arid zones. Excellent ground cover that prevents wind erosion."
    },
    "mungbean": {
        "common_name": "Mung Bean (Green Gram / Moong)",
        "category": "Pulse / Legume",
        "season": "Kharif / Zaid (Summer)",
        "soil_type": "Well-drained fertile loam to sandy loam (pH 6.2 - 7.2)",
        "growth_duration": "60 - 75 days",
        "water_requirement": "Low to Moderate (Sensitive to excess water and overcast weather)",
        "fertilizer_guide": "15-20 kg N and 40 kg P2O5; benefits greatly from phosphorus fertilization.",
        "economic_value": "Short duration cash turnaround; high nutrition and sprout demand.",
        "description": "Rapid-growth legume ideal for crop rotation between main cereal seasons. Matures in barely two months."
    },
    "blackgram": {
        "common_name": "Black Gram (Urad Dal)",
        "category": "Pulse / Legume",
        "season": "Kharif / Spring",
        "soil_type": "Heavier soils, loamy or clay loam with high organic content",
        "growth_duration": "70 - 85 days",
        "water_requirement": "Moderate (600-750 mm rainfall)",
        "fertilizer_guide": "Starter Nitrogen (20 kg/ha), Phosphorus (40 kg/ha), Sulfur supplementation.",
        "economic_value": "Essential for culinary batters (dosa, idli, dal makhani); consistently high market rate.",
        "description": "Nutrient-dense legume rich in phosphoric acid. Tolerates warm humid climates and enhances soil organic nitrogen."
    },
    "lentil": {
        "common_name": "Lentil (Masoor Dal)",
        "category": "Pulse / Legume",
        "season": "Rabi (Cool winter season)",
        "soil_type": "Alluvial, light loams, clay soils with moderate drainage",
        "growth_duration": "110 - 130 days",
        "water_requirement": "Low (Grown primarily as rainfed or with 1-2 irrigations)",
        "fertilizer_guide": "20 kg N, 40 kg P2O5, 20 kg K2O per hectare.",
        "economic_value": "Global pulse commodity; stable consumer demand and low input cost.",
        "description": "Cool-season legume with delicate foliage. Highly efficient in low-moisture winter conditions."
    },
    "pomegranate": {
        "common_name": "Pomegranate (Anar)",
        "category": "Horticulture / Fruit",
        "season": "Perennial (Flowers: Ambe, Mrig, Hasta bahar)",
        "soil_type": "Deep well-drained sandy loam or alluvial soil (tolerates slight salinity)",
        "growth_duration": "Perennial (bearing from 2nd-3rd year onward)",
        "water_requirement": "Moderate (Drip irrigation recommended; dry weather needed during ripening)",
        "fertilizer_guide": "High Potassium during fruit development; balanced N-P-K (600:200:200 g/plant/year).",
        "economic_value": "High-value export fruit; strong domestic antioxidant beverage and fresh fruit market.",
        "description": "Subtropical fruit renowned for drought tolerance and high antioxidant arils. Prefers semi-arid climates with warm summers."
    },
    "banana": {
        "common_name": "Banana",
        "category": "Horticulture / Fruit",
        "season": "Year-round planting in tropical zones",
        "soil_type": "Rich, deep, loose loamy soil with high organic matter (pH 6.5 - 7.5)",
        "growth_duration": "11 - 14 months to harvest",
        "water_requirement": "Very High (1800 - 2500 mm or regular drip irrigation every 3-4 days)",
        "fertilizer_guide": "Heavy potassium and nitrogen feeder (200g N, 60g P, 300g K per plant).",
        "economic_value": "Highest return per acre among quick fruit crops with continuous harvest cycles.",
        "description": "Fast-growing giant herbaceous perennial requiring tropical heat, heavy moisture, and substantial nutrient supply."
    },
    "mango": {
        "common_name": "Mango",
        "category": "Horticulture / Fruit",
        "season": "Summer harvest (Spring flowering)",
        "soil_type": "Deep, well-drained alluvial, red, or lateritic loam (pH 5.5 - 7.5)",
        "growth_duration": "Perennial orchard (commercial harvest from 4th-5th year)",
        "water_requirement": "Moderate (Stress required before flowering; regular irrigation during fruit set)",
        "fertilizer_guide": "Graduated NPK dosage per tree age; micro-nutrients (Boron, Zinc) prevent fruit drop.",
        "economic_value": "'King of Fruits' with exceptional domestic demand and lucrative international export.",
        "description": "Tropical fruit with extensive lifespan. Requires distinct dry spell in winter to stimulate blossom induction."
    },
    "grapes": {
        "common_name": "Grapes",
        "category": "Horticulture / Fruit",
        "season": "Perennial vine (Pruning: Oct-Nov, Harvest: Mar-May)",
        "soil_type": "Sandy loam, gravelly well-drained soils with no hard pan",
        "growth_duration": "Perennial vineyard (bearing for 15-20+ years)",
        "water_requirement": "Moderate (Precise drip management; dry sunny weather during harvest)",
        "fertilizer_guide": "Very high Potassium (K) requirement for sugar translocation and bunch quality.",
        "economic_value": "Commercial powerhouse for table consumption, raisins, juice, and viticulture.",
        "description": "Vine crop thriving in semi-arid sunny conditions with low humidity during berry ripening to prevent fungal mildew."
    },
    "watermelon": {
        "common_name": "Watermelon",
        "category": "Cucurbit / Fruit",
        "season": "Zaid / Summer (Late spring to mid-summer)",
        "soil_type": "Sandy, sandy loam, warm riverbed soils with rapid drainage",
        "growth_duration": "80 - 100 days",
        "water_requirement": "Moderate (High early on; reduce watering near harvest to sweeten brix)",
        "fertilizer_guide": "Balanced NPK (80:50:50 kg/ha); avoid excess late Nitrogen which softens rinds.",
        "economic_value": "High quick-turn profit crop during peak hot summer months.",
        "description": "Warm-season vine requiring long sunny days and warm soil. Sensitive to cold temperatures and stagnant water."
    },
    "muskmelon": {
        "common_name": "Muskmelon (Cantaloupe / Kharbooza)",
        "category": "Cucurbit / Fruit",
        "season": "Zaid / Summer",
        "soil_type": "Fertile sandy loam with good drainage and pH 6.0 - 7.0",
        "growth_duration": "75 - 90 days",
        "water_requirement": "Moderate (Irrigate furrow or drip; keep vines dry to prevent mildew)",
        "fertilizer_guide": "NPK (60:40:40 kg/ha); potassium enhances aromatics and netting quality.",
        "economic_value": "High demand in domestic fruit markets during scorching summer periods.",
        "description": "Thrives in dry, hot summer conditions. High sunshine and low humidity during ripening create maximum sweetness and aroma."
    },
    "apple": {
        "common_name": "Apple",
        "category": "Temperate Fruit",
        "season": "Temperate (Harvest: Aug-Oct)",
        "soil_type": "Deep, loamy, well-aerated soil rich in humus (pH 5.5 - 6.5)",
        "growth_duration": "Perennial orchard (deciduous tree)",
        "water_requirement": "Moderate (1000-1250 mm distributed rainfall / drip)",
        "fertilizer_guide": "High Potassium and Phosphorus; requires adequate Calcium to prevent bitter pit.",
        "economic_value": "Premium cash crop in temperate hill regions with long cold-storage viability.",
        "description": "Temperate deciduous fruit requiring 1,000-1,500 chilling hours below 7°C to break winter dormancy."
    },
    "orange": {
        "common_name": "Orange (Citrus / Mandarin)",
        "category": "Citrus / Fruit",
        "season": "Winter / Spring harvest",
        "soil_type": "Well-drained light loamy or sandy loam (pH 6.0 - 7.5, zero waterlogging)",
        "growth_duration": "Perennial orchard (bearing for 25-30 years)",
        "water_requirement": "Moderate (Regular scheduled drip; sensitive to salinity and stagnant water)",
        "fertilizer_guide": "Balanced citrus nutrition (600g N, 200g P2O5, 400g K2O/plant); Micronutrients: Zn, Fe, Mn.",
        "economic_value": "Perennial cash crop with strong beverage, processing, and fresh fruit value.",
        "description": "Subtropical citrus tree demanding sunny climate, well-aerated root zone, and moderate humidity."
    },
    "papaya": {
        "common_name": "Papaya",
        "category": "Horticulture / Fruit",
        "season": "Year-round in warm tropical & frost-free zones",
        "soil_type": "Rich, well-drained alluvial or loam soil with high humus content",
        "growth_duration": "9 - 12 months to first harvest; productive for 2-3 years",
        "water_requirement": "High to Moderate (Highly vulnerable to collar rot if flooded)",
        "fertilizer_guide": "Heavy feeder: 200g N, 200g P, 250g K per plant applied in 6 bi-monthly splits.",
        "economic_value": "Fast capital return; high consumer demand for fresh consumption and papain enzyme.",
        "description": "Fast-growing tropical tree-like herb producing fruit within a year. Extremely sensitive to frost and water accumulation."
    },
    "coconut": {
        "common_name": "Coconut",
        "category": "Plantation Crop",
        "season": "Perennial (Continuous monthly nut harvests)",
        "soil_type": "Coastal sandy, alluvial, or red sandy loam with good depth",
        "growth_duration": "Perennial (economic yield starts year 5-7, lasts 60+ years)",
        "water_requirement": "High (1500 - 2500 mm rainfall or continuous subsoil coastal moisture)",
        "fertilizer_guide": "High Potassium (K) and Chlorine (500g N, 320g P, 1200g K per palm/year).",
        "economic_value": "'Kalpavriksha' (Tree of Life); every part produces income (water, copra, oil, coir).",
        "description": "The quintessential tropical palm thriving along coastal belts and humid lowlands with abundant sun and moisture."
    },
    "cotton": {
        "common_name": "Cotton (White Gold)",
        "category": "Fiber / Cash Crop",
        "season": "Kharif (Sown: April-June, Picked: Oct-Jan)",
        "soil_type": "Deep black cotton soils (regur) or fertile loams with high clay retention",
        "growth_duration": "150 - 180 days",
        "water_requirement": "Moderate (700-1100 mm; dry weather essential during boll opening)",
        "fertilizer_guide": "NPK (100:50:50 kg/ha); magnesium and boron prevent leaf reddening and boll drop.",
        "economic_value": "Foremost natural textile fiber worldwide; critical export commodity.",
        "description": "Major commercial fiber crop requiring warm climate, at least 180-200 frost-free days, and sunny boll-maturing season."
    },
    "jute": {
        "common_name": "Jute (Golden Fiber)",
        "category": "Fiber / Cash Crop",
        "season": "Kharif (Sown: March-May)",
        "soil_type": "New alluvial soils deposited by annual river flooding (pH 6.0 - 7.5)",
        "growth_duration": "120 - 150 days",
        "water_requirement": "Very High (Thrives in monsoon floodplains; standing water tolerated late stage)",
        "fertilizer_guide": "High Nitrogen (60-80 kg N/ha) for rapid vegetative stem elongation; 30 kg P, 30 kg K.",
        "economic_value": "Eco-friendly biodegradable packaging, burlap, geotextiles, and specialty paper.",
        "description": "The world's leading golden fiber crop, flourishing in the Bengal and Brahmaputra river basins with monsoon heat and humidity."
    },
    "coffee": {
        "common_name": "Coffee (Arabica / Robusta)",
        "category": "Plantation / Beverage",
        "season": "Perennial Hill Crop (Blossom showers in March; Harvest: Nov-Feb)",
        "soil_type": "Deep, porous, well-drained volcanic or forest loam rich in organic humus (pH 6.0 - 6.5)",
        "growth_duration": "Perennial shrub (bearing for 30-50 years)",
        "water_requirement": "High to Moderate (1500 - 2000 mm with distinct dry winter period before blossom)",
        "fertilizer_guide": "Balanced NPK (120:90:120 kg/ha) with trace elements, shade tree organic mulch.",
        "economic_value": "Top global traded beverage commodity with high international market premium.",
        "description": "Shade-grown tropical highland shrub cultivated on misty hill slopes. Requires filtered sunlight and cool hill air."
    }
}

def train_and_export():
    print("Loading data from:", DATA_PATH)
    df = pd.read_csv(DATA_PATH)
    
    feature_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    X = df[feature_cols]
    y = df['label']
    
    print(f"Dataset shape: {X.shape}, Classes: {len(y.unique())}")
    
    # Stratified Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Compare 3 Models
    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=80, max_depth=16, random_state=42)
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    comparison_results = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        test_acc = accuracy_score(y_test, preds)
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        comparison_results[name] = {
            "test_accuracy": round(float(test_acc) * 100, 2),
            "cv_mean_accuracy": round(float(cv_scores.mean()) * 100, 2),
            "cv_std": round(float(cv_scores.std()) * 100, 3)
        }
        print(f"[{name}] Test Accuracy: {test_acc*100:.2f}%, 5-Fold CV Mean: {cv_scores.mean()*100:.2f}%")
        
    best_model = models["Random Forest"]
    best_preds = best_model.predict(X_test)
    f1 = f1_score(y_test, best_preds, average='weighted')
    report = classification_report(y_test, best_preds, output_dict=True)
    
    # Feature importances
    importances = {
        col: round(float(imp) * 100, 2)
        for col, imp in zip(feature_cols, best_model.feature_importances_)
    }
    # Sort descending
    importances = dict(sorted(importances.items(), key=lambda item: item[1], reverse=True))
    
    # Save scikit-learn compressed model
    joblib.dump(best_model, MODEL_JOBLIB_PATH, compress=3)
    print(f"Saved joblib model to {MODEL_JOBLIB_PATH} ({os.path.getsize(MODEL_JOBLIB_PATH) / 1024:.1f} KB)")
    
    # Export pure-JSON model representation for zero-dependency inference
    forest_data = {
        'n_classes': len(best_model.classes_),
        'classes': best_model.classes_.tolist(),
        'n_features': len(feature_cols),
        'feature_names': feature_cols,
        'trees': []
    }
    
    for estimator in best_model.estimators_:
        t = estimator.tree_
        forest_data['trees'].append({
            'children_left': t.children_left.tolist(),
            'children_right': t.children_right.tolist(),
            'feature': t.feature.tolist(),
            'threshold': [round(x, 4) for x in t.threshold.tolist()],
            'value': [[round(x, 4) for x in v[0].tolist()] for v in t.value]
        })
        
    with open(MODEL_JSON_PATH, 'w') as f:
        json.dump(forest_data, f)
    print(f"Saved JSON tree model to {MODEL_JSON_PATH} ({os.path.getsize(MODEL_JSON_PATH) / 1024:.1f} KB)")
    
    # Compute per-crop statistics from dataset
    crop_profiles = {}
    for crop, group in df.groupby('label'):
        details = CROP_DETAILS.get(crop, {
            "common_name": crop.capitalize(),
            "category": "Agricultural Crop",
            "season": "Seasonal",
            "soil_type": "Loam / Sandy Loam",
            "growth_duration": "90 - 120 days",
            "water_requirement": "Moderate",
            "fertilizer_guide": "Balanced NPK",
            "economic_value": "Commercial and subsistence value.",
            "description": f"{crop.capitalize()} is grown under suitable agro-climatic conditions."
        })
        
        crop_profiles[crop] = {
            "name": crop,
            **details,
            "stats": {
                "N": {"mean": round(group['N'].mean(), 1), "min": float(group['N'].min()), "max": float(group['N'].max())},
                "P": {"mean": round(group['P'].mean(), 1), "min": float(group['P'].min()), "max": float(group['P'].max())},
                "K": {"mean": round(group['K'].mean(), 1), "min": float(group['K'].min()), "max": float(group['K'].max())},
                "temperature": {"mean": round(group['temperature'].mean(), 1), "min": round(float(group['temperature'].min()), 1), "max": round(float(group['temperature'].max()), 1)},
                "humidity": {"mean": round(group['humidity'].mean(), 1), "min": round(float(group['humidity'].min()), 1), "max": round(float(group['humidity'].max()), 1)},
                "ph": {"mean": round(group['ph'].mean(), 2), "min": round(float(group['ph'].min()), 2), "max": round(float(group['ph'].max()), 2)},
                "rainfall": {"mean": round(group['rainfall'].mean(), 1), "min": round(float(group['rainfall'].min()), 1), "max": round(float(group['rainfall'].max()), 1)}
            }
        }
        
    with open(METADATA_PATH, 'w') as f:
        json.dump(crop_profiles, f, indent=2)
    print(f"Saved crop metadata for {len(crop_profiles)} crops to {METADATA_PATH}")
    
    # Overall dataset ranges for UI sliders and boundary validation
    dataset_ranges = {}
    for col in feature_cols:
        dataset_ranges[col] = {
            "min": float(df[col].min()),
            "max": float(df[col].max()),
            "mean": round(float(df[col].mean()), 1),
            "median": round(float(df[col].median()), 1)
        }
        
    metrics_summary = {
        "model_name": "Random Forest Ensemble (80 Trees, Max Depth 16)",
        "test_accuracy": round(float(accuracy_score(y_test, best_preds)) * 100, 2),
        "f1_score": round(float(f1) * 100, 2),
        "total_samples": int(len(df)),
        "num_crops": len(best_model.classes_),
        "crops": best_model.classes_.tolist(),
        "feature_importances": importances,
        "model_comparison": comparison_results,
        "dataset_ranges": dataset_ranges
    }
    
    with open(METRICS_PATH, 'w') as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"Saved metrics summary to {METRICS_PATH}")
    
    print("\nTraining completed successfully! Production assets ready.")

if __name__ == '__main__':
    train_and_export()
