import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
import plotly.express as px
import plotly.graph_objects as go
import shap
import matplotlib.pyplot as plt
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
    .metric-card {
        background: linear-gradient(135deg, #1e2130, #2d3250);
        border-radius: 12px; padding: 20px;
        border-left: 4px solid #e74c3c;
        margin: 8px 0;
    }
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
st.markdown("**ADB AI for Safer Roads Innovation Challenge** | By Muhammad Abdul Qadeer — Qadeer Automations")
st.markdown("---")

# ─── SIDEBAR ───────────────────────────────────────────────
st.sidebar.image("https://www.adb.org/sites/default/files/adb-logo.png", width=120)
st.sidebar.title("⚙️ Controls")
st.sidebar.markdown("---")

data_source = st.sidebar.radio(
    "📂 Data Source",
    ["Use Demo Data", "Upload CSV"],
    help="Upload real ADB dataset or use synthetic demo"
)

# ─── FEATURE ENGINEERING FUNCTION ──────────────────────────
def engineer_features(df):
    df = df.copy()

    df['speed_deviation_ratio'] = df['speed_85th_percentile'] / df['posted_speed_limit']

    road_type_expected = {
        'highway': 100, 'arterial': 60,
        'collector': 50, 'local': 40, 'residential': 30
    }
    df['expected_speed'] = df['road_type'].map(road_type_expected).fillna(60)
    df['limit_mismatch'] = abs(df['posted_speed_limit'] - df['expected_speed'])
    df['overlimit_ratio'] = df['posted_speed_limit'] / df['expected_speed'].replace(0, 1)

    df['school_risk']   = np.where(df['proximity_to_school_m'] < 500, 2.0, 1.0)
    df['market_risk']   = np.where(df['proximity_to_market_m'] < 300, 1.5, 1.0)
    df['urban_risk']    = df['urban_rural'].map({'urban': 2.0, 'suburban': 1.5, 'rural': 1.0}).fillna(1.5)
    df['land_use_risk'] = df['land_use_type'].map({
        'school_zone': 3.0, 'market': 2.5,
        'residential': 2.0, 'industrial': 1.5, 'open_road': 1.0
    }).fillna(1.5)

    df['vulnerability_index'] = (
        df['school_risk']         * 0.25 +
        df['market_risk']         * 0.20 +
        df['urban_risk']          * 0.20 +
        df['land_use_risk']       * 0.15 +
        df['ptwheeler_indicator'] * 0.10 +
        df.get('cyclist_exposure', pd.Series(0.5, index=df.index)) * 0.10
    )

    safe_limits = {'urban': 50, 'suburban': 80, 'rural': 100}
    df['safe_limit'] = df['urban_rural'].map(safe_limits).fillna(60)
    df['safe_system_violation'] = (df['posted_speed_limit'] > df['safe_limit']).astype(int)

    scaler = MinMaxScaler()
    df['norm_speed_deviation'] = scaler.fit_transform(df[['speed_deviation_ratio']])
    df['norm_vulnerability']   = scaler.fit_transform(df[['vulnerability_index']])
    df['norm_intersection']    = scaler.fit_transform(df[['intersection_density']])
    df['norm_population']      = scaler.fit_transform(df[['population_density']])
    df['norm_limit_mismatch']  = scaler.fit_transform(df[['limit_mismatch']])

    df['speed_safety_score'] = (
        df['norm_speed_deviation']  * 0.35 +
        df['norm_vulnerability']    * 0.25 +
        df['norm_limit_mismatch']   * 0.20 +
        df['norm_intersection']     * 0.10 +
        df['norm_population']       * 0.10
    ) * 100

    df['risk_category'] = pd.cut(
        df['speed_safety_score'],
        bins=[0, 25, 50, 75, 100],
        labels=['Low Risk', 'Moderate Risk', 'High Risk', 'Critical Risk']
    )
    df['is_unsafe'] = (df['speed_safety_score'] > 50).astype(int)

    return df

# ─── SYNTHETIC DATA ─────────────────────────────────────────
@st.cache_data
def generate_demo_data():
    np.random.seed(42)
    n = 1000
    return pd.DataFrame({
        'segment_id':            [f'SEG_{i:04d}' for i in range(n)],
        'country':               np.random.choice(['Pakistan','Philippines','Indonesia','Bangladesh','Vietnam'], n),
        'latitude':              np.random.uniform(10.0, 35.0, n),
        'longitude':             np.random.uniform(70.0, 130.0, n),
        'posted_speed_limit':    np.random.choice([30,40,50,60,80,100,120], n),
        'operating_speed_mean':  np.random.uniform(20, 120, n),
        'speed_85th_percentile': np.random.uniform(30, 130, n),
        'traffic_intensity':     np.random.uniform(100, 5000, n),
        'road_type':             np.random.choice(['highway','arterial','collector','local','residential'], n),
        'urban_rural':           np.random.choice(['urban','suburban','rural'], n),
        'intersection_density':  np.random.uniform(0, 20, n),
        'segment_length_km':     np.random.uniform(0.1, 5.0, n),
        'land_use_type':         np.random.choice(['school_zone','market','residential','industrial','open_road'], n),
        'population_density':    np.random.uniform(50, 15000, n),
        'proximity_to_school_m': np.random.uniform(50, 5000, n),
        'proximity_to_market_m': np.random.uniform(50, 5000, n),
        'ptwheeler_indicator':   np.random.uniform(0, 1, n),
        'cyclist_exposure':      np.random.uniform(0, 1, n),
    })

# ─── LOAD DATA ──────────────────────────────────────────────
if data_source == "Upload CSV":
    uploaded = st.sidebar.file_uploader("Upload ADB Dataset (CSV)", type=["csv"])
    if uploaded:
        raw_df = pd.read_csv(uploaded)
        st.sidebar.success(f"✅ Loaded {len(raw_df)} segments")
    else:
        st.sidebar.info("⬆️ Upload CSV to begin")
        st.info("👈 Upload your ADB dataset from the sidebar, or switch to Demo Data")
        st.stop()
else:
    raw_df = generate_demo_data()
    st.sidebar.success("✅ Demo data loaded (1000 segments)")

# ─── PROCESS DATA ───────────────────────────────────────────
with st.spinner("🔄 Running SpeedSafe AI pipeline..."):
    df = engineer_features(raw_df)

# ─── SIDEBAR FILTERS ────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.subheader("🔍 Filters")

countries = ['All'] + sorted(df['country'].unique().tolist()) if 'country' in df.columns else ['All']
sel_country = st.sidebar.selectbox("Country", countries)

risk_filter = st.sidebar.multiselect(
    "Risk Category",
    ['Low Risk', 'Moderate Risk', 'High Risk', 'Critical Risk'],
    default=['High Risk', 'Critical Risk']
)

score_range = st.sidebar.slider("Safety Score Range", 0, 100, (0, 100))

# Apply filters
fdf = df.copy()
if sel_country != 'All':
    fdf = fdf[fdf['country'] == sel_country]
if risk_filter:
    fdf = fdf[fdf['risk_category'].isin(risk_filter)]
fdf = fdf[(fdf['speed_safety_score'] >= score_range[0]) &
          (fdf['speed_safety_score'] <= score_range[1])]

# ─── KPI METRICS ────────────────────────────────────────────
st.subheader("📊 Key Risk Indicators")
col1, col2, col3, col4, col5 = st.columns(5)

total       = len(df)
critical    = len(df[df['risk_category'] == 'Critical Risk'])
high        = len(df[df['risk_category'] == 'High Risk'])
violations  = df['safe_system_violation'].sum()
avg_score   = df['speed_safety_score'].mean()

col1.metric("🛣️ Total Segments",     f"{total:,}")
col2.metric("🔴 Critical Risk",       f"{critical:,}", f"{critical/total*100:.1f}%")
col3.metric("🟠 High Risk",           f"{high:,}",     f"{high/total*100:.1f}%")
col4.metric("⚠️ Safe System Violations", f"{int(violations):,}")
col5.metric("📈 Avg Safety Score",    f"{avg_score:.1f}/100")

st.markdown("---")

# ─── TABS ───────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🗺️ Interactive Map",
    "📊 Risk Dashboard",
    "🤖 ML Model",
    "📋 Data Explorer"
])

# ══════════════════════════════════════════════════════════
# TAB 1 — MAP
# ══════════════════════════════════════════════════════════
with tab1:
    st.subheader("🗺️ Speed-Unsafe Segments Map")

    map_type = st.radio("Map Type", ["Risk Points", "Heatmap", "Both"], horizontal=True)

    color_map = {
        'Low Risk':      '#2ecc71',
        'Moderate Risk': '#f39c12',
        'High Risk':     '#e67e22',
        'Critical Risk': '#e74c3c'
    }

    m = folium.Map(location=[20.0, 100.0], zoom_start=4, tiles='CartoDB dark_matter')

    if map_type in ["Risk Points", "Both"]:
        layers = {
            'Critical Risk': folium.FeatureGroup(name='🔴 Critical Risk'),
            'High Risk':     folium.FeatureGroup(name='🟠 High Risk'),
            'Moderate Risk': folium.FeatureGroup(name='🟡 Moderate Risk'),
            'Low Risk':      folium.FeatureGroup(name='🟢 Low Risk'),
        }
        for _, row in fdf.iterrows():
            cat = str(row['risk_category'])
            if cat in layers:
                folium.CircleMarker(
                    location=[row['latitude'], row['longitude']],
                    radius=7 if cat == 'Critical Risk' else 5,
                    color=color_map.get(cat, 'gray'),
                    fill=True, fill_opacity=0.85,
                    popup=folium.Popup(f"""
                        <div style='font-family:Arial;font-size:12px;width:220px'>
                        <h4 style='color:#e74c3c;margin:0'>⚠️ {cat}</h4><hr>
                        <b>Segment:</b> {row['segment_id']}<br>
                        <b>Country:</b> {row.get('country','N/A')}<br>
                        <b>Safety Score:</b> <b style='color:#e74c3c'>{row['speed_safety_score']:.1f}/100</b><br>
                        <b>Posted Limit:</b> {row['posted_speed_limit']} km/h<br>
                        <b>85th Pct Speed:</b> {row['speed_85th_percentile']:.1f} km/h<br>
                        <b>Road Type:</b> {row['road_type']}<br>
                        <b>Area:</b> {row['urban_rural']}<br>
                        <b>Safe System:</b> {'⚠️ VIOLATION' if row['safe_system_violation'] else '✅ OK'}
                        </div>
                    """, max_width=250)
                ).add_to(layers[cat])
        for layer in layers.values():
            layer.add_to(m)

    if map_type in ["Heatmap", "Both"]:
        crit = df[df['risk_category'] == 'Critical Risk']
        heat_data = [[r['latitude'], r['longitude'], r['speed_safety_score']/100]
                     for _, r in crit.iterrows()]
        HeatMap(heat_data, name='🔥 Risk Heatmap',
                min_opacity=0.4, radius=20, blur=15).add_to(m)

    legend_html = """
    <div style="position:fixed;bottom:30px;left:30px;z-index:1000;
         background:rgba(0,0,0,0.85);padding:15px;border-radius:10px;
         color:white;font-size:13px;font-family:Arial;border:1px solid #555">
    <b style='font-size:14px'>🚦 Speed Safety Score</b><br><br>
    <span style='color:#2ecc71'>●</span> Low Risk (0–25)<br>
    <span style='color:#f39c12'>●</span> Moderate Risk (25–50)<br>
    <span style='color:#e67e22'>●</span> High Risk (50–75)<br>
    <span style='color:#e74c3c'>●</span> Critical Risk (75–100)<br><br>
    <i style='font-size:10px'>SpeedSafe AI | Qadeer Automations</i>
    </div>"""
    m.get_root().html.add_child(folium.Element(legend_html))
    folium.LayerControl(collapsed=False).add_to(m)

    st_folium(m, width=None, height=550)
    st.caption(f"Showing {len(fdf):,} segments after filters")

# ══════════════════════════════════════════════════════════
# TAB 2 — DASHBOARD
# ══════════════════════════════════════════════════════════
with tab2:
    st.subheader("📊 Risk Analysis Dashboard")

    col1, col2 = st.columns(2)

    with col1:
        risk_counts = df['risk_category'].value_counts().reset_index()
        risk_counts.columns = ['Category', 'Count']
        fig1 = px.pie(risk_counts, values='Count', names='Category',
                      color='Category',
                      color_discrete_map={
                          'Low Risk':'#2ecc71','Moderate Risk':'#f39c12',
                          'High Risk':'#e67e22','Critical Risk':'#e74c3c'
                      },
                      title='Risk Category Distribution')
        fig1.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='white')
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        fig2 = px.histogram(df, x='speed_safety_score', nbins=40,
                            color_discrete_sequence=['#e74c3c'],
                            title='Speed Safety Score Distribution',
                            labels={'speed_safety_score': 'Safety Score (0=Safe, 100=Critical)'})
        fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='white')
        st.plotly_chart(fig2, use_container_width=True)

    if 'country' in df.columns:
        country_stats = df.groupby('country').agg(
            avg_score=('speed_safety_score','mean'),
            critical=('risk_category', lambda x: (x=='Critical Risk').sum()),
            violations=('safe_system_violation','sum')
        ).reset_index()

        fig3 = px.bar(country_stats, x='country', y='avg_score',
                      color='critical', color_continuous_scale='Reds',
                      title='Average Safety Score by Country',
                      text='avg_score',
                      labels={'avg_score':'Avg Score','critical':'Critical Segments'})
        fig3.update_traces(texttemplate='%{text:.1f}', textposition='outside')
        fig3.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='white', height=400)
        st.plotly_chart(fig3, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        road_risk = df.groupby('road_type')['speed_safety_score'].mean().reset_index()
        fig4 = px.bar(road_risk, x='road_type', y='speed_safety_score',
                      color='speed_safety_score', color_continuous_scale='RdYlGn_r',
                      title='Avg Safety Score by Road Type')
        fig4.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='white')
        st.plotly_chart(fig4, use_container_width=True)

    with col4:
        urban_risk = df.groupby('urban_rural')['speed_safety_score'].mean().reset_index()
        fig5 = px.bar(urban_risk, x='urban_rural', y='speed_safety_score',
                      color='speed_safety_score', color_continuous_scale='RdYlGn_r',
                      title='Avg Safety Score by Area Type')
        fig5.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='white')
        st.plotly_chart(fig5, use_container_width=True)

# ══════════════════════════════════════════════════════════
# TAB 3 — ML MODEL
# ══════════════════════════════════════════════════════════
with tab3:
    st.subheader("🤖 Ensemble ML Model — XGBoost + Random Forest + Gradient Boosting")

    features = [
        'speed_deviation_ratio','speed_85th_percentile','posted_speed_limit',
        'traffic_intensity','intersection_density','segment_length_km',
        'vulnerability_index','population_density','ptwheeler_indicator',
        'limit_mismatch','overlimit_ratio','safe_system_violation'
    ]
    features = [f for f in features if f in df.columns]

    X = df[features]
    y = df['is_unsafe']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with st.spinner("Training ensemble model..."):
        xgb_m = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                                    random_state=42, eval_metric='logloss', verbosity=0)
        rf_m  = RandomForestClassifier(n_estimators=100, random_state=42)
        gb_m  = GradientBoostingClassifier(n_estimators=100, random_state=42)

        ensemble = VotingClassifier(
            estimators=[('xgb',xgb_m),('rf',rf_m),('gb',gb_m)],
            voting='soft'
        )
        ensemble.fit(X_train, y_train)
        y_pred = ensemble.predict(X_test)
        y_prob = ensemble.predict_proba(X_test)[:,1]
        auc    = roc_auc_score(y_test, y_prob)

    col1, col2, col3 = st.columns(3)
    col1.metric("🎯 ROC-AUC Score", f"{auc:.4f}")
    col2.metric("🧠 Model Type", "Ensemble (3 models)")
    col3.metric("📦 Training Samples", f"{len(X_train):,}")

    st.markdown("#### 📋 Classification Report")
    report = classification_report(y_test, y_pred,
                                   target_names=['Safe','Unsafe'],
                                   output_dict=True)
    report_df = pd.DataFrame(report).transpose().round(3)
    st.dataframe(report_df, use_container_width=True)

    # SHAP
    st.markdown("#### 🔍 SHAP Feature Importance — Why is a segment unsafe?")
    with st.spinner("Calculating SHAP values..."):
        xgb_m.fit(X_train, y_train)
        explainer   = shap.TreeExplainer(xgb_m)
        shap_values = explainer.shap_values(X_test)

    fig_shap, ax = plt.subplots(figsize=(10, 6))
    fig_shap.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')
    shap.summary_plot(shap_values, X_test, plot_type='bar',
                      show=False, color='#e74c3c')
    plt.title('Feature Importance (SHAP)', color='white', fontsize=14)
    plt.tick_params(colors='white')
    st.pyplot(fig_shap)

# ══════════════════════════════════════════════════════════
# TAB 4 — DATA EXPLORER
# ══════════════════════════════════════════════════════════
with tab4:
    st.subheader("📋 Road Segment Data Explorer")

    show_cols = ['segment_id','country','speed_safety_score','risk_category',
                 'posted_speed_limit','speed_85th_percentile','road_type',
                 'urban_rural','land_use_type','safe_system_violation','vulnerability_index']
    show_cols = [c for c in show_cols if c in df.columns]

    sort_col = st.selectbox("Sort by", ['speed_safety_score','vulnerability_index','limit_mismatch'])
    top_n    = st.slider("Show top N segments", 10, 200, 50)

    display_df = df[show_cols].sort_values(sort_col, ascending=False).head(top_n)
    st.dataframe(display_df, use_container_width=True, height=450)

    csv = df[show_cols].to_csv(index=False).encode('utf-8')
    st.download_button(
        "⬇️ Download Full Results CSV",
        csv,
        "SpeedSafe_Results.csv",
        "text/csv"
    )

# ─── FOOTER ─────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center;color:#666;font-size:12px'>
🚦 SpeedSafe AI | Muhammad Abdul Qadeer | Qadeer Automations<br>
ADB AI for Safer Roads Innovation Challenge 2026
</div>
""", unsafe_allow_html=True)
