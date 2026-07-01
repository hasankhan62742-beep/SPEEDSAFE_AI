import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import plotly.express as px
import requests
import tempfile
import os
import warnings
warnings.filterwarnings('ignore')

# ─── PAGE CONFIG ───────────────────────────────────────────
st.set_page_config(
    page_title="SpeedSafe AI | ADB Challenge",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CUSTOM CSS ────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .hero-title {
        font-size: 2.5rem; font-weight: 800;
        background: linear-gradient(90deg, #e74c3c, #f39c12);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .stMetric { background: #1e2130; border-radius: 10px; padding: 10px; }
</style>
""", unsafe_allow_html=True)

# ─── HEADER ────────────────────────────────────────────────
st.markdown('<p class="hero-title">🚦 SpeedSafe AI</p>', unsafe_allow_html=True)
st.markdown("**ADB AI for Safer Roads Innovation Challenge 2026** | Muhammad Abdul Qadeer — Qadeer Automations")
st.markdown("---")

# ─── GITHUB RELEASE URLs ───────────────────────────────────
THAILAND_URL    = "https://github.com/hasankhan62742-beep/SPEEDSAFE_AI/releases/download/v1.0-data/ADB_Innovation_Thailand.geojson"
MAHARASHTRA_URL = "https://github.com/hasankhan62742-beep/SPEEDSAFE_AI/releases/download/v1.0-data/ADB_Innovation_Maharashtra.geojson"

# ─── DOWNLOAD + CACHE REAL DATA ────────────────────────────
@st.cache_data(show_spinner=False)
def download_geojson(url, name):
    try:
        with st.spinner(f"⬇️ Loading {name} real data..."):
            r = requests.get(url, timeout=120)
            r.raise_for_status()
            with tempfile.NamedTemporaryFile(suffix='.geojson', delete=False) as f:
                f.write(r.content)
                return gpd.read_file(f.name)
    except Exception as e:
        st.error(f"❌ Could not load {name}: {e}")
        return None

# ─── FEATURE ENGINEERING ───────────────────────────────────
def process_real_data(gdf, country_name):
    df = gdf.copy()
    df['country']   = country_name
    df['longitude'] = df.geometry.centroid.x
    df['latitude']  = df.geometry.centroid.y

    df['posted_speed_limit']    = pd.to_numeric(df.get('SpeedLimit'), errors='coerce')
    df['speed_85th_percentile'] = pd.to_numeric(df.get('F85thPercentileSpeed'), errors='coerce')
    df['operating_speed_mean']  = pd.to_numeric(df.get('MedianSpeed'), errors='coerce')
    df['traffic_intensity']     = pd.to_numeric(df.get('WeightedSample'), errors='coerce')
    df['road_type']             = df.get('RoadClass', pd.Series('primary', index=df.index)).str.lower().fillna('primary')
    df['percent_over_limit']    = pd.to_numeric(df.get('PercentOverLimit'), errors='coerce').fillna(0)
    df['segment_length_km']     = pd.to_numeric(df.get('Shape_Length'), errors='coerce').fillna(1) / 1000
    df['road_name']             = df.get('english_ro', pd.Series('Unknown', index=df.index)).fillna('Unknown')

    if 'LandUse' in df.columns:
        df['urban_rural'] = df['LandUse'].str.upper().map({'URBAN':'urban','RURAL':'rural'}).fillna('rural')
    else:
        df['urban_rural'] = 'rural'

    df = df.dropna(subset=['posted_speed_limit','speed_85th_percentile'])
    df = df[df['posted_speed_limit'] > 0]

    # Feature Engineering
    df['speed_deviation_ratio'] = df['speed_85th_percentile'] / df['posted_speed_limit']
    road_expected = {'motorway':100,'trunk':80,'primary':60,'secondary':50}
    df['expected_speed']  = df['road_type'].map(road_expected).fillna(60)
    df['limit_mismatch']  = abs(df['posted_speed_limit'] - df['expected_speed'])
    safe_limits = {'urban':50,'rural':100}
    df['safe_limit']            = df['urban_rural'].map(safe_limits).fillna(80)
    df['safe_system_violation'] = (df['posted_speed_limit'] > df['safe_limit']).astype(int)
    df['urban_risk'] = df['urban_rural'].map({'urban':2.0,'rural':1.0}).fillna(1.5)
    df['road_risk']  = df['road_type'].map({'motorway':1.0,'trunk':1.5,'primary':2.0,'secondary':2.5}).fillna(1.5)
    df['vulnerability_index'] = df['urban_risk']*0.5 + df['road_risk']*0.5

    scaler = MinMaxScaler()
    for col in ['speed_deviation_ratio','vulnerability_index','limit_mismatch','percent_over_limit']:
        df[f'norm_{col}'] = scaler.fit_transform(df[[col]])

    df['speed_safety_score'] = (
        df['norm_speed_deviation_ratio'] * 0.35 +
        df['norm_vulnerability_index']   * 0.25 +
        df['norm_limit_mismatch']        * 0.20 +
        df['norm_percent_over_limit']    * 0.20
    ) * 100

    df['risk_category'] = pd.cut(
        df['speed_safety_score'],
        bins=[0,25,50,75,100],
        labels=['Low Risk','Moderate Risk','High Risk','Critical Risk']
    )
    df['is_unsafe'] = (df['speed_safety_score'] > 50).astype(int)
    return df

# ─── SIDEBAR ───────────────────────────────────────────────
st.sidebar.title("⚙️ Controls")
st.sidebar.markdown("---")

data_source = st.sidebar.radio(
    "📂 Data Source",
    ["🌍 Real ADB Data (Thailand + Maharashtra)", "📂 Upload Your Own CSV"],
)

# ─── LOAD DATA ─────────────────────────────────────────────
if data_source == "🌍 Real ADB Data (Thailand + Maharashtra)":
    col1, col2 = st.columns(2)
    with col1:
        with st.spinner("⬇️ Loading Thailand real data (82MB)..."):
            th_gdf = download_geojson(THAILAND_URL, "Thailand")
    with col2:
        with st.spinner("⬇️ Loading Maharashtra real data (41MB)..."):
            mh_gdf = download_geojson(MAHARASHTRA_URL, "Maharashtra")

    if th_gdf is None or mh_gdf is None:
        st.error("❌ Could not load real data. Please try uploading CSV instead.")
        st.stop()

    with st.spinner("🔄 Processing 70,000+ road segments..."):
        th_df = process_real_data(th_gdf, "Thailand")
        mh_df = process_real_data(mh_gdf, "Maharashtra")
        df    = pd.concat([th_df, mh_df], ignore_index=True)

    st.success(f"✅ Real ADB Data Loaded — {len(df):,} road segments!")

else:
    uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        raw = pd.read_csv(uploaded)
        # Map uploaded CSV columns to standard names
        col_map = {
            'SpeedLimit':'posted_speed_limit',
            'F85thPercentileSpeed':'speed_85th_percentile',
            'MedianSpeed':'operating_speed_mean',
            'WeightedSample':'traffic_intensity',
            'RoadClass':'road_type',
            'PercentOverLimit':'percent_over_limit',
            'LandUse':'urban_rural',
            'Shape_Length':'segment_length_km',
            'english_ro':'road_name'
        }
        raw = raw.rename(columns={k:v for k,v in col_map.items() if k in raw.columns})
        raw['country'] = raw.get('country', 'Unknown')
        if 'latitude' not in raw.columns:
            st.error("❌ CSV must have latitude/longitude columns!")
            st.stop()
        raw = raw.dropna(subset=['posted_speed_limit','speed_85th_percentile'])
        raw['speed_deviation_ratio'] = raw['speed_85th_percentile'] / raw['posted_speed_limit']
        raw['road_type'] = raw.get('road_type', 'primary').fillna('primary').str.lower()
        road_expected = {'motorway':100,'trunk':80,'primary':60,'secondary':50}
        raw['expected_speed'] = raw['road_type'].map(road_expected).fillna(60)
        raw['limit_mismatch'] = abs(raw['posted_speed_limit'] - raw['expected_speed'])
        raw['urban_rural'] = raw.get('urban_rural','rural').str.lower().fillna('rural')
        safe_limits = {'urban':50,'rural':100}
        raw['safe_limit'] = raw['urban_rural'].map(safe_limits).fillna(80)
        raw['safe_system_violation'] = (raw['posted_speed_limit'] > raw['safe_limit']).astype(int)
        raw['urban_risk'] = raw['urban_rural'].map({'urban':2.0,'rural':1.0}).fillna(1.5)
        raw['road_risk']  = raw['road_type'].map({'motorway':1.0,'trunk':1.5,'primary':2.0,'secondary':2.5}).fillna(1.5)
        raw['vulnerability_index'] = raw['urban_risk']*0.5 + raw['road_risk']*0.5
        raw['percent_over_limit']  = pd.to_numeric(raw.get('percent_over_limit', 0), errors='coerce').fillna(0)
        scaler = MinMaxScaler()
        for col in ['speed_deviation_ratio','vulnerability_index','limit_mismatch','percent_over_limit']:
            raw[f'norm_{col}'] = scaler.fit_transform(raw[[col]])
        raw['speed_safety_score'] = (
            raw['norm_speed_deviation_ratio'] * 0.35 +
            raw['norm_vulnerability_index']   * 0.25 +
            raw['norm_limit_mismatch']        * 0.20 +
            raw['norm_percent_over_limit']    * 0.20
        ) * 100
        raw['risk_category'] = pd.cut(raw['speed_safety_score'],
            bins=[0,25,50,75,100],
            labels=['Low Risk','Moderate Risk','High Risk','Critical Risk'])
        raw['is_unsafe'] = (raw['speed_safety_score'] > 50).astype(int)
        df = raw
        st.success(f"✅ CSV Loaded — {len(df):,} segments!")
    else:
        st.info("👈 Upload a CSV from the sidebar, or switch to Real ADB Data")
        st.stop()

# ─── SIDEBAR FILTERS ───────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.subheader("🔍 Filters")
countries  = ['All'] + sorted(df['country'].unique().tolist())
sel_country = st.sidebar.selectbox("Country", countries)
risk_filter = st.sidebar.multiselect(
    "Risk Category",
    ['Low Risk','Moderate Risk','High Risk','Critical Risk'],
    default=['High Risk','Critical Risk']
)
score_range = st.sidebar.slider("Safety Score Range", 0, 100, (0,100))

fdf = df.copy()
if sel_country != 'All':
    fdf = fdf[fdf['country'] == sel_country]
if risk_filter:
    fdf = fdf[fdf['risk_category'].isin(risk_filter)]
fdf = fdf[(fdf['speed_safety_score'] >= score_range[0]) &
          (fdf['speed_safety_score'] <= score_range[1])]

# ─── KPI METRICS ───────────────────────────────────────────
st.subheader("📊 Key Risk Indicators")
c1,c2,c3,c4,c5 = st.columns(5)
total      = len(df)
critical   = (df['risk_category']=='Critical Risk').sum()
high       = (df['risk_category']=='High Risk').sum()
violations = df['safe_system_violation'].sum()
avg_score  = df['speed_safety_score'].mean()

c1.metric("🛣️ Total Segments",        f"{total:,}")
c2.metric("🔴 Critical Risk",          f"{critical:,}", f"{critical/total*100:.1f}%")
c3.metric("🟠 High Risk",              f"{high:,}",     f"{high/total*100:.1f}%")
c4.metric("⚠️ Safe System Violations", f"{int(violations):,}")
c5.metric("📈 Avg Safety Score",       f"{avg_score:.1f}/100")
st.markdown("---")

# ─── TABS ──────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🗺️ Interactive Map","📊 Risk Dashboard","🤖 ML Model","📋 Data Explorer"
])

# ══ TAB 1 — MAP ════════════════════════════════════════════
with tab1:
    st.subheader("🗺️ Speed-Unsafe Segments — Real Road Network")
    map_type = st.radio("Map Type", ["Risk Points","Heatmap","Both"], horizontal=True)

    color_map = {
        'Low Risk':'#2ecc71','Moderate Risk':'#f39c12',
        'High Risk':'#e67e22','Critical Risk':'#e74c3c'
    }

    # Sample for performance if too many points
    map_df = fdf if len(fdf) <= 5000 else fdf.sample(5000, random_state=42)
    if len(fdf) > 5000:
        st.info(f"ℹ️ Showing 5,000 of {len(fdf):,} filtered segments for map performance")

    center_lat = map_df['latitude'].mean() if len(map_df) > 0 else 15.0
    center_lon = map_df['longitude'].mean() if len(map_df) > 0 else 100.0

    m = folium.Map(location=[center_lat, center_lon], zoom_start=5,
                   tiles='CartoDB dark_matter')

    if map_type in ["Risk Points","Both"]:
        layers = {
            'Critical Risk': folium.FeatureGroup(name='🔴 Critical Risk'),
            'High Risk':     folium.FeatureGroup(name='🟠 High Risk'),
            'Moderate Risk': folium.FeatureGroup(name='🟡 Moderate Risk'),
            'Low Risk':      folium.FeatureGroup(name='🟢 Low Risk'),
        }
        for _, row in map_df.iterrows():
            cat = str(row['risk_category'])
            if cat not in color_map: continue
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=7 if cat=='Critical Risk' else 5,
                color=color_map[cat], fill=True, fill_opacity=0.85,
                popup=folium.Popup(f"""
                    <div style='font-family:Arial;font-size:12px;width:230px'>
                    <h4 style='color:#e74c3c;margin:0'>⚠️ {cat}</h4><hr>
                    <b>Road:</b> {row.get('road_name','N/A')}<br>
                    <b>Country:</b> {row['country']}<br>
                    <b>Safety Score:</b> <b style='color:#e74c3c'>{row['speed_safety_score']:.1f}/100</b><br>
                    <b>Posted Limit:</b> {row['posted_speed_limit']} km/h<br>
                    <b>85th Pct Speed:</b> {row['speed_85th_percentile']:.1f} km/h<br>
                    <b>Road Class:</b> {row['road_type']}<br>
                    <b>Area:</b> {row['urban_rural']}<br>
                    <b>Safe System:</b> {'⚠️ VIOLATION' if row['safe_system_violation'] else '✅ OK'}
                    </div>
                """, max_width=260)
            ).add_to(layers[cat])
        for layer in layers.values():
            layer.add_to(m)

    if map_type in ["Heatmap","Both"]:
        high_risk = map_df[map_df['risk_category'].isin(['High Risk','Critical Risk'])]
        heat_data = [[r['latitude'],r['longitude'],r['speed_safety_score']/100]
                     for _,r in high_risk.iterrows()]
        if heat_data:
            HeatMap(heat_data, name='🔥 Risk Heatmap',
                    min_opacity=0.5, radius=25, blur=20).add_to(m)

    legend_html = """
    <div style="position:fixed;bottom:30px;left:30px;z-index:1000;
         background:rgba(0,0,0,0.85);padding:15px;border-radius:10px;
         color:white;font-size:13px;font-family:Arial;border:1px solid #555">
    <b style='font-size:14px'>🚦 Speed Safety Score</b><br><br>
    <span style='color:#2ecc71'>●</span> Low Risk (0–25)<br>
    <span style='color:#f39c12'>●</span> Moderate Risk (25–50)<br>
    <span style='color:#e67e22'>●</span> High Risk (50–75)<br>
    <span style='color:#e74c3c'>●</span> Critical Risk (75–100)<br><br>
    <i style='font-size:10px'>SpeedSafe AI | Qadeer Automations<br>ADB Challenge 2026</i>
    </div>"""
    m.get_root().html.add_child(folium.Element(legend_html))
    folium.LayerControl(collapsed=False).add_to(m)
    st_folium(m, width=None, height=550)
    st.caption(f"Showing {len(map_df):,} segments | Filtered: {len(fdf):,} total")

# ══ TAB 2 — DASHBOARD ══════════════════════════════════════
with tab2:
    st.subheader("📊 Risk Analysis Dashboard")
    c1, c2 = st.columns(2)
    with c1:
        rc = df['risk_category'].value_counts().reset_index()
        rc.columns = ['Category','Count']
        fig1 = px.pie(rc, values='Count', names='Category',
                      color='Category',
                      color_discrete_map={'Low Risk':'#2ecc71','Moderate Risk':'#f39c12',
                                          'High Risk':'#e67e22','Critical Risk':'#e74c3c'},
                      title='Risk Category Distribution')
        fig1.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='white')
        st.plotly_chart(fig1, use_container_width=True)
    with c2:
        fig2 = px.histogram(df, x='speed_safety_score', nbins=40,
                            color_discrete_sequence=['#e74c3c'],
                            title='Speed Safety Score Distribution',
                            labels={'speed_safety_score':'Safety Score (0=Safe, 100=Critical)'})
        fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='white')
        st.plotly_chart(fig2, use_container_width=True)

    country_stats = df.groupby('country').agg(
        avg_score=('speed_safety_score','mean'),
        critical=('risk_category', lambda x:(x=='Critical Risk').sum()),
        violations=('safe_system_violation','sum'),
        total=('speed_safety_score','count')
    ).reset_index()
    country_stats['violation_pct'] = (country_stats['violations']/country_stats['total']*100).round(1)
    fig3 = px.bar(country_stats, x='country', y='avg_score',
                  color='critical', color_continuous_scale='Reds',
                  title='Average Safety Score by Country',
                  text='violation_pct',
                  labels={'avg_score':'Avg Score','critical':'Critical Segments'})
    fig3.update_traces(texttemplate='%{text}% violations', textposition='outside')
    fig3.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='white', height=400)
    st.plotly_chart(fig3, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        road_risk = df.groupby('road_type')['speed_safety_score'].mean().reset_index()
        fig4 = px.bar(road_risk, x='road_type', y='speed_safety_score',
                      color='speed_safety_score', color_continuous_scale='RdYlGn_r',
                      title='Avg Safety Score by Road Type')
        fig4.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='white')
        st.plotly_chart(fig4, use_container_width=True)
    with c4:
        urban_risk = df.groupby('urban_rural')['speed_safety_score'].mean().reset_index()
        fig5 = px.bar(urban_risk, x='urban_rural', y='speed_safety_score',
                      color='speed_safety_score', color_continuous_scale='RdYlGn_r',
                      title='Avg Safety Score by Area Type')
        fig5.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='white')
        st.plotly_chart(fig5, use_container_width=True)

# ══ TAB 3 — ML MODEL ═══════════════════════════════════════
with tab3:
    st.subheader("🤖 Ensemble ML Model — XGBoost + Random Forest + Gradient Boosting")
    features = [f for f in [
        'speed_deviation_ratio','speed_85th_percentile','posted_speed_limit',
        'traffic_intensity','limit_mismatch','vulnerability_index',
        'safe_system_violation','percent_over_limit'
    ] if f in df.columns]

    X = df[features].fillna(0)
    y = df['is_unsafe']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with st.spinner("🧠 Training ensemble model on real ADB data..."):
        xgb_m = xgb.XGBClassifier(n_estimators=100, max_depth=6, random_state=42,
                                    eval_metric='logloss', verbosity=0)
        rf_m  = RandomForestClassifier(n_estimators=100, random_state=42)
        gb_m  = GradientBoostingClassifier(n_estimators=100, random_state=42)
        ensemble = VotingClassifier(
            estimators=[('xgb',xgb_m),('rf',rf_m),('gb',gb_m)], voting='soft')
        ensemble.fit(X_train, y_train)
        y_pred = ensemble.predict(X_test)
        y_prob = ensemble.predict_proba(X_test)[:,1]
        auc    = roc_auc_score(y_test, y_prob)

    c1,c2,c3 = st.columns(3)
    c1.metric("🎯 ROC-AUC Score",    f"{auc:.4f}")
    c2.metric("🧠 Model Type",       "Ensemble (3 models)")
    c3.metric("📦 Training Samples", f"{len(X_train):,}")

    st.markdown("#### 📋 Classification Report")
    report    = classification_report(y_test, y_pred,
                    target_names=['Safe','Unsafe'], output_dict=True)
    report_df = pd.DataFrame(report).transpose().round(3)
    st.dataframe(report_df, use_container_width=True)

    st.markdown("#### 🔍 SHAP — Why Is a Segment Unsafe?")
    with st.spinner("Calculating SHAP values..."):
        xgb_m.fit(X_train, y_train)
        explainer   = shap.TreeExplainer(xgb_m)
        shap_values = explainer.shap_values(X_test)
    fig_shap, ax = plt.subplots(figsize=(10,6))
    fig_shap.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')
    shap.summary_plot(shap_values, X_test, plot_type='bar', show=False, color='#e74c3c')
    plt.title('Feature Importance (SHAP) — Real ADB Data', color='white', fontsize=13)
    plt.tick_params(colors='white')
    st.pyplot(fig_shap)

# ══ TAB 4 — DATA EXPLORER ══════════════════════════════════
with tab4:
    st.subheader("📋 Road Segment Explorer")
    show_cols = [c for c in [
        'road_name','country','speed_safety_score','risk_category',
        'posted_speed_limit','speed_85th_percentile','road_type',
        'urban_rural','safe_system_violation','vulnerability_index'
    ] if c in df.columns]

    sort_col = st.selectbox("Sort by", ['speed_safety_score','vulnerability_index','limit_mismatch'])
    top_n    = st.slider("Show top N segments", 10, 500, 50)

    display_df = df[show_cols].sort_values(sort_col, ascending=False).head(top_n)
    st.dataframe(display_df, use_container_width=True, height=450)

    csv = df[show_cols].to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download Full Results CSV", csv,
                       "SpeedSafe_Results.csv", "text/csv")

# ─── FOOTER ────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center;color:#666;font-size:12px'>
🚦 SpeedSafe AI | Muhammad Abdul Qadeer | Qadeer Automations | Lahore, Pakistan<br>
ADB AI for Safer Roads Innovation Challenge 2026
</div>
""", unsafe_allow_html=True)
