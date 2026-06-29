# 🚦 SpeedSafe AI — ADB AI for Safer Roads Innovation Challenge 2026

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-Live-red?style=for-the-badge&logo=streamlit)
![XGBoost](https://img.shields.io/badge/XGBoost-Ensemble-green?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

> **Can AI tell us where speed limits need to be updated to better protect road users?**
> 
> This project answers that question using machine learning, geospatial analysis, and mobility data.

---

## 🌏 About This Project

**SpeedSafe AI** is a submission for the [ADB AI for Safer Roads Innovation Challenge 2026](https://challenges.adb.org/en/challenges/ai4saferroads), led by the Asian Development Bank in collaboration with the World Bank, AI for Good, and ITU.

The challenge asks: *How might we use AI and mobility data to determine where speed limits are misaligned with real-world road conditions, supporting evidence-based speed management across Asia and the Pacific?*

This is **not** about measuring whether drivers are speeding.  
This is about determining whether the **speed limit itself is appropriate** for the road.

---

## 🎯 What SpeedSafe AI Does

| Capability | Description |
|-----------|-------------|
| ✅ Safe Speed Assessment | Evaluates whether posted speed limits align with road function, operating speeds, and land use |
| ✅ Risk Identification | Identifies segments exposing pedestrians, cyclists, and motorcyclists to unacceptable risk |
| ✅ Speed Safety Score | 0–100 composite score per road segment (higher = more unsafe) |
| ✅ Geospatial Visualization | Interactive map of Speed-Unsafe Segments with color-coded risk layers |
| ✅ Policy-Ready Output | Country-level summaries governments can act on immediately |
| ✅ Scalable Methodology | Replicable across any ADB member country with GPS probe data |

---

## 🗺️ Live Demo

🔗 **[Launch SpeedSafe AI App](https://speedsafe-ai.streamlit.app)**

Features:
- 🗺️ Interactive dark-theme map with clickable road segments
- 🔥 Risk heatmap overlay
- 📊 Real-time dashboard with country & road-type analysis
- 🤖 Ensemble ML model (XGBoost + Random Forest + Gradient Boosting)
- 🔍 SHAP explainability — why each segment is flagged
- 📂 Upload your own CSV dataset for instant analysis

---

## 📊 Methodology

### 1. Data Sources
- **GPS Probe Data**: Operating speeds, 85th percentile speeds, posted limits, traffic intensity
- **Road Network Data**: Functional class, urban/rural classification, intersection density, segment length
- **Contextual Layers**: Population density, land use, proximity to schools/markets, powered two-wheeler indicators
- **Mapillary Imagery**: ML-identified road features and signs

### 2. Feature Engineering

```python
# Key engineered features:
speed_deviation_ratio  = speed_85th_percentile / posted_speed_limit
limit_mismatch         = abs(posted_speed_limit - expected_speed_for_road_type)
vulnerability_index    = weighted(school_risk, market_risk, urban_risk, land_use_risk, ptw_exposure)
safe_system_violation  = posted_speed_limit > safe_system_threshold(urban/suburban/rural)
```

### 3. Speed Safety Score Formula

```
Speed Safety Score (0–100) =
    Speed Deviation     × 0.35  +
    Vulnerability Index × 0.25  +
    Limit Mismatch      × 0.20  +
    Intersection Risk   × 0.10  +
    Population Density  × 0.10
```

### 4. Risk Categories

| Score | Category | Action |
|-------|----------|--------|
| 0–25 | 🟢 Low Risk | Monitor |
| 25–50 | 🟡 Moderate Risk | Review |
| 50–75 | 🟠 High Risk | Prioritize |
| 75–100 | 🔴 Critical Risk | Immediate Intervention |

### 5. ML Model

**Ensemble Voting Classifier** (soft voting):
- XGBoost (n=200, depth=6)
- Random Forest (n=200, depth=8)
- Gradient Boosting (n=150, depth=5)

**Explainability**: SHAP (SHapley Additive exPlanations) values for every prediction

### 6. Safe System Compliance Check

Based on WHO Safe System principles:
- Urban roads: ≤ 50 km/h
- Suburban roads: ≤ 80 km/h
- Rural roads: ≤ 100 km/h

Any segment exceeding these thresholds is flagged as a **Safe System Violation**.

---

## 📁 Repository Structure

```
speedsafe-ai/
│
├── app.py                    # Streamlit web application
├── requirements.txt          # Python dependencies
├── README.md                 # This file
│
├── notebooks/
│   └── ADB_SpeedSafe_AI.ipynb   # Full analysis notebook (Google Colab)
│
├── outputs/
│   ├── SpeedSafe_AI_Map.html    # Standalone interactive map
│   ├── SpeedSafe_Results.csv    # Scored road segments
│   ├── score_distribution.png   # Risk score chart
│   └── shap_importance.png      # Feature importance chart
│
└── docs/
    └── Findings_Summary.docx    # Policy report (max 5 pages)
```

---

## 🚀 Run Locally

```bash
# Clone the repo
git clone https://github.com/hasankhan62742-beep/speedsafe-ai.git
cd speedsafe-ai

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

---

## 📦 Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.10+ | Core language |
| Streamlit | Web application |
| XGBoost + Scikit-learn | ML models |
| SHAP | Model explainability |
| Folium | Interactive maps |
| GeoPandas | Geospatial processing |
| Plotly | Interactive charts |
| Pandas / NumPy | Data processing |

---

## 🌍 Scalability

This methodology is designed to be **country-agnostic**. Any nation with GPS probe data and road network data can apply SpeedSafe AI. The pipeline:

1. Accepts any CSV with standard columns
2. Auto-detects road types and urban/rural classification
3. Applies Safe System thresholds appropriate for the region
4. Produces a country-level risk summary for policymakers

Currently demonstrated across: **Pakistan, Philippines, Indonesia, Bangladesh, Vietnam**

---

## 👤 Team

**Muhammad Abdul Qadeer**  
AI/ML Engineer | Qadeer Automations  
📍 Lahore, Pakistan  
🔗 [LinkedIn](https://linkedin.com/in/muhammad-abdul-qadeer-8262b237b) | [GitHub](https://github.com/hasankhan62742-beep)

---

## 📄 License

MIT License — free to use, adapt, and deploy for road safety purposes.

---

## 🤝 Acknowledgements

- Asian Development Bank (ADB) — Challenge organizers
- World Bank Development Impact Group — Partners
- AI for Good / ITU — Partners
- Mapillary — Street-level imagery data

---

*Submitted to the ADB AI for Safer Roads Innovation Challenge 2026*  
*Goal: Evidence-based speed management across Asia and the Pacific* 🌏
