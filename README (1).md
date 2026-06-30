# 🚦 SpeedSafe AI — ADB AI for Safer Roads Innovation Challenge 2026

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-Live-red?style=for-the-badge&logo=streamlit)
![XGBoost](https://img.shields.io/badge/XGBoost-Ensemble-green?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

> **Can AI tell us where speed limits need to be updated to better protect road users?**
>
> This project answers that question using machine learning, geospatial analysis, and real mobility data from Thailand and Maharashtra, India.

---

## 🌏 About This Project

**SpeedSafe AI** is a submission for the [ADB AI for Safer Roads Innovation Challenge 2026](https://challenges.adb.org/en/challenges/ai4saferroads), led by the Asian Development Bank in collaboration with the World Bank, AI for Good, and ITU.

This is **not** about measuring whether drivers are speeding.
This is about determining whether the **speed limit itself is appropriate** for the road.

---

## 🔗 Live Links

| Resource | Link |
|---|---|
| 🗺️ **Live Streamlit App** | [speedsafe-fkmqptbsczewxgzzzp3vlk.streamlit.app](https://speedsafe-fkmqptbsczewxgzzzp3vlk.streamlit.app/) |
| 📓 **Full Analysis Notebook (Google Colab)** | [Open in Colab](https://colab.research.google.com/drive/1aRzVewaNEId3xU8k6jnlafvm4xqNKfp1?usp=sharing) |
| 📄 **Findings Summary Report** | [docs/SpeedSafe_Findings_Summary.docx](docs/SpeedSafe_Findings_Summary.docx) |

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

## 📊 Key Findings (Real ADB Data)

- **14,711** real road segments analyzed across Thailand and Maharashtra
- **5,679** Safe System violations identified (38.6% of network)
- **320** High Risk segments flagged for immediate intervention
- **99.6% ROC-AUC** achieved by the ensemble ML model (5-fold cross-validated)
- Most dangerous segment found: **Srinagarindra-Rom Klao Road, Bangkok** — posted 30 km/h, actual 85th-percentile speed of 91–95 km/h

---

## 📊 Methodology

### 1. Data Sources
- **GPS Probe Data**: Operating speeds, 85th percentile speeds, posted limits, traffic intensity (TomTom)
- **Road Network Data**: Functional class, urban/rural classification, segment length (Overture Maps / OSM)
- **Land Use**: NASA GRUMP dataset for urban/rural classification

### 2. Feature Engineering

```
speed_deviation_ratio  = speed_85th_percentile / posted_speed_limit
limit_mismatch         = abs(posted_speed_limit - expected_speed_for_road_class)
vulnerability_index    = urban_risk * 0.5 + road_class_risk * 0.5
safe_system_violation  = posted_speed_limit > safe_system_threshold(urban/rural)
```

### 3. Speed Safety Score Formula

```
Speed Safety Score (0–100) =
    Speed Deviation     × 0.35  +
    Vulnerability Index × 0.25  +
    Limit Mismatch      × 0.20  +
    % Over Limit         × 0.20
```

### 4. Risk Categories

| Score | Category | Action |
|-------|----------|--------|
| 0–25 | 🟢 Low Risk | Monitor |
| 25–50 | 🟡 Moderate Risk | Review |
| 50–75 | 🟠 High Risk | Prioritize |
| 75–100 | 🔴 Critical Risk | Immediate Intervention |

### 5. Machine Learning Model

**Ensemble Voting Classifier** (soft voting) trained on real ADB data:
- XGBoost (n=200, depth=6)
- Random Forest (n=200, depth=8)
- Gradient Boosting (n=150, depth=5)

**Result:** 99.6% ROC-AUC, 5-fold cross-validated, with SHAP explainability for every prediction.

### 6. Safe System Compliance Check

Based on WHO Safe System principles:
- Urban roads: ≤ 50 km/h
- Rural roads: ≤ 100 km/h

Any segment exceeding these thresholds is flagged as a **Safe System Violation**.

---

## 📁 Repository Structure

```
speedsafe-ai/
│
├── app.py                            # Streamlit web application
├── requirements.txt                  # Python dependencies
├── README.md                         # This file
│
├── docs/
│   └── SpeedSafe_Findings_Summary.docx   # Policy report (max 5 pages)
│
└── (Full analysis notebook hosted on Google Colab — see Live Links above)
```

---

## 🚀 Run Locally

```bash
git clone https://github.com/hasankhan62742-beep/SPEEDSAFE_AI.git
cd SPEEDSAFE_AI
pip install -r requirements.txt
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

SpeedSafe AI requires only three data inputs — posted speed limits, GPS probe speeds, and road class — all available via TomTom, HERE, or OpenStreetMap in most ADB member countries. The pipeline auto-adapts Safe System thresholds by urban/rural classification and can be deployed in any country within weeks of data access.

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
- TomTom, Overture Maps Foundation, NASA GRUMP — Data sources

---

*Submitted to the ADB AI for Safer Roads Innovation Challenge 2026*
*Goal: Evidence-based speed management across Asia and the Pacific* 🌏
