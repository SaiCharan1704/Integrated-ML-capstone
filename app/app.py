import os
import sys
from pathlib import Path
import datetime
import numpy as np
import pandas as pd
import streamlit as st
import joblib

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Traffic Intelligence · Team Cobra Tech",
    page_icon="🐍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for clean modern presentation
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #34f089;
        margin-bottom: 0px;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #94a3b8;
        margin-bottom: 15px;
    }
    .metric-card {
        background-color: #0f1d33;
        border: 1px solid #1e293b;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding-top: 10px;
        padding-bottom: 10px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# PATH CONFIGURATION & CACHED MODEL LOADERS
# -----------------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
MODELS_DIR = ROOT_DIR / "models"

@st.cache_resource(show_spinner="Loading Regression Model...")
def load_regression_bundle():
    bundle_path = MODELS_DIR / "regression_bundle.joblib"
    if bundle_path.exists():
        return joblib.load(bundle_path)
    return None

@st.cache_resource(show_spinner="Loading Classification Model...")
def load_classification_bundle():
    bundle_path = MODELS_DIR / "classification_bundle.joblib"
    if bundle_path.exists():
        return joblib.load(bundle_path)
    return None

@st.cache_resource(show_spinner="Loading Clustering Model...")
def load_clustering_bundle():
    bundle_path = MODELS_DIR / "clustering_bundle.joblib"
    if bundle_path.exists():
        return joblib.load(bundle_path)
    return None

reg_bundle = load_regression_bundle()
clf_bundle = load_classification_bundle()
cluster_bundle = load_clustering_bundle()

# -----------------------------------------------------------------------------
# SIDEBAR
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.shields.io/badge/Team-Cobra%20Tech-34f089?style=for-the-badge&labelColor=060d1a", use_container_width=True)
    st.markdown("### 🎓 **Machine Learning Capstone**")
    st.markdown("**Course:** 23CSE301 · III Year B.Tech CSE")
    st.markdown("**Project:** End-to-End Traffic Intelligence System")
    
    st.markdown("---")
    st.markdown("### 👥 **Team Members**")
    st.markdown("""
    - **K. Sai charan**  
      *(Track 1: Regression Lead)*
    - **L. Hrudhay**  
      *(Track 2: Classification Lead)*
    - **K. Sapthagiri Reddy**  
      *(Track 3: Clustering Lead)*
    """)
    
    st.markdown("---")
    st.markdown("### 🛠️ **System Architecture**")
    st.markdown("""
    - **22 ML Algorithms** evaluated
    - **Leakage-safe** `scikit-learn` pipelines
    - **Reproducible seed:** `42`
    - **Deployment:** Streamlit Cloud
    """)

# -----------------------------------------------------------------------------
# HEADER
# -----------------------------------------------------------------------------
st.markdown('<div class="main-title">🐍 Team Cobra Tech — Traffic Intelligence System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Real-Time Interactive Prediction & Spatial Analytics Dashboard across 3 Machine Learning Tracks</div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# MAIN NAVIGATION TABS
# -----------------------------------------------------------------------------
tab_reg, tab_clf, tab_cluster, tab_score, tab_about = st.tabs([
    "📈 Track 1 · Traffic Volume (Regression)",
    "🚨 Track 2 · Accident Severity (Classification)",
    "🗺️ Track 3 · GPS Trajectory (Clustering)",
    "🏆 Model Leaderboard & Metrics",
    "ℹ️ Project Architecture & Team"
])

# =============================================================================
# TAB 1: REGRESSION TRACK
# =============================================================================
with tab_reg:
    st.markdown("### 📈 Highway Traffic Volume Prediction (I-94 Interstate)")
    st.markdown("Predict hourly traffic volume using meteorological data, time features, and national holidays. Fitted on **48,204 records**.")
    
    if reg_bundle is None:
        st.error("⚠️ Regression model bundle not found in `models/`. Please run `python export_models.py` first.")
    else:
        reg_pipe = reg_bundle["pipeline"]
        reg_meta = reg_bundle["metadata"]
        
        col_input, col_pred = st.columns([1.1, 0.9], gap="large")
        
        with col_input:
            st.markdown("#### ⚙️ Input Conditions")
            
            c1, c2 = st.columns(2)
            with c1:
                input_date = st.date_input("Date", datetime.date(2018, 9, 30))
            with c2:
                input_time = st.time_input("Time of Day", datetime.time(16, 0))
            
            # Weather details
            c3, c4 = st.columns(2)
            with c3:
                temp_c = st.slider("Temperature (°C)", min_value=-30.0, max_value=38.0, value=20.0, step=0.5, help="Recorded historical range in Minneapolis I-94: -29.8°C to +36.9°C")
                temp_k = temp_c + 273.15
            with c4:
                clouds = st.slider("Cloud Cover (%)", min_value=0, max_value=100, value=40, step=5)
            
            c5, c6 = st.columns(2)
            with c5:
                rain_1h = st.number_input("Rainfall in last hour (mm)", min_value=0.0, max_value=100.0, value=0.0, step=0.1)
            with c6:
                snow_1h = st.number_input("Snowfall in last hour (mm)", min_value=0.0, max_value=50.0, value=0.0, step=0.1)
            
            c7, c8 = st.columns(2)
            with c7:
                weather_main = st.selectbox("General Weather", reg_meta["weather_mains"], index=reg_meta["weather_mains"].index("Clouds") if "Clouds" in reg_meta["weather_mains"] else 0)
            with c8:
                holiday = st.selectbox("Holiday Status", reg_meta["holidays"], index=reg_meta["holidays"].index("None") if "None" in reg_meta["holidays"] else 0)
            
            # Filter weather description based on weather main if available
            filtered_descs = [d for d in reg_meta["weather_descs"] if weather_main.lower() in d.lower()]
            if not filtered_descs:
                filtered_descs = reg_meta["weather_descs"]
            weather_desc = st.selectbox("Detailed Weather Description", filtered_descs)
            
            # Derived temporal features
            hour = input_time.hour
            day_of_week = input_date.weekday()
            month = input_date.month
            year = input_date.year
            is_weekend = 1 if day_of_week >= 5 else 0
            
            if 0 <= hour <= 5:
                traffic_period = "Night"
            elif 6 <= hour <= 11:
                traffic_period = "Morning"
            elif 12 <= hour <= 17:
                traffic_period = "Afternoon"
            else:
                traffic_period = "Evening"
            
            reg_input_df = pd.DataFrame([{
                "temp": temp_k,
                "rain_1h": rain_1h,
                "snow_1h": snow_1h,
                "clouds_all": clouds,
                "hour": hour,
                "day_of_week": day_of_week,
                "month": month,
                "year": year,
                "is_weekend": is_weekend,
                "holiday": holiday,
                "weather_main": weather_main,
                "weather_description": weather_desc,
                "traffic_period": traffic_period
            }])
        
        with col_pred:
            st.markdown("#### 🎯 Prediction Results")
            predicted_volume = float(reg_pipe.predict(reg_input_df)[0])
            predicted_volume = max(0.0, predicted_volume)
            
            st.metric(
                label="Predicted Hourly Traffic Volume",
                value=f"{int(predicted_volume):,} vehicles / hr",
                delta=f"± 230 vehicles (MAE Bound)"
            )
            
            # Level assessment
            if predicted_volume < 1500:
                level_color = "🟢"
                level_desc = "Light Traffic (Free Flow)"
                alert_type = st.success
            elif predicted_volume < 3500:
                level_color = "🟡"
                level_desc = "Moderate Flow"
                alert_type = st.info
            elif predicted_volume < 5000:
                level_color = "🟠"
                level_desc = "Heavy Traffic"
                alert_type = st.warning
            else:
                level_color = "🔴"
                level_desc = "Severe Peak Congestion"
                alert_type = st.error
                
            alert_type(f"**Flow State:** {level_color} **{level_desc}**")
            
            st.markdown("##### 🔍 Real-Time Feature Analysis:")
            st.markdown(f"""
            - **Time Slot:** `{hour:02d}:00` · Period: **{traffic_period}**
            - **Day:** `{input_date.strftime('%A')}` ({'Weekend' if is_weekend else 'Weekday'})
            - **Weather Impact:** `{weather_main}` ({temp_c:.1f}°C)
            - **Selected Algorithm:** Tuned Random Forest ($R^2 = 0.956$, RMSE $= 418.7$)
            """)
            
            st.markdown("##### 💡 Model Insight:")
            st.caption("Feature importance analysis confirmed that **hour and traffic period account for ~84% of variance**. Weekdays exhibit a classic bimodal peak (07:00 and 16:00), while weekends exhibit a single midday plateau.")
            if temp_c <= -15.0:
                st.info("❄️ **Real-World Minnesota Insight:** The dataset comes from Minneapolis I-94. Even during extreme sub-zero weather (-20°C to -29°C), historical records show over 850 hours of active traffic (averaging 2,479 vehicles/hr and up to 6,900 at 4 PM rush hour). People still commute to work in heated cars and snow-plowed freeways!")

# =============================================================================
# TAB 2: CLASSIFICATION TRACK
# =============================================================================
with tab_clf:
    st.markdown("### 🚨 Road Traffic Accident (RTA) Severity Classification")
    st.markdown("Predict the accident severity level (**Slight Injury**, **Serious Injury**, or **Fatal Injury**) based on driver demographics, road geography, and incident conditions.")
    
    if clf_bundle is None:
        st.error("⚠️ Classification model bundle not found in `models/`. Please run `python export_models.py` first.")
    else:
        clf_pipe = clf_bundle["pipeline"]
        clf_meta = clf_bundle["metadata"]
        cat_opts = clf_meta["categorical_options"]
        
        col_c_in, col_c_out = st.columns([1.1, 0.9], gap="large")
        
        with col_c_in:
            st.markdown("#### 📋 Incident Parameters")
            
            with st.expander("👤 Driver & Vehicle Profile", expanded=True):
                ca1, ca2 = st.columns(2)
                with ca1:
                    age_band = st.selectbox("Driver Age Band", cat_opts.get("Age_band_of_driver", ["18-30", "31-50", "Over 51", "Under 18"]))
                    sex_driver = st.selectbox("Driver Gender", cat_opts.get("Sex_of_driver", ["Male", "Female", "Unknown"]))
                    edu_level = st.selectbox("Driver Education", cat_opts.get("Educational_level", ["Elementary school", "Junior high school", "High school", "Above high school"]))
                with ca2:
                    driving_exp = st.selectbox("Driving Experience", cat_opts.get("Driving_experience", ["1-2yr", "2-5yr", "5-10yr", "Above 10yr", "No Licence"]))
                    vehicle_type = st.selectbox("Vehicle Type", cat_opts.get("Type_of_vehicle", ["Automobile", "Public (> 45 seats)", "Lorry (41?100Q)", "Pick up upto 10Q"]))
                    vehicle_defect = st.selectbox("Vehicle Defect", cat_opts.get("Defect_of_vehicle", ["No defect", "7", "Defective brakes"]))
            
            with st.expander("🛣️ Road & Environmental Conditions", expanded=True):
                cb1, cb2 = st.columns(2)
                with cb1:
                    road_surface = st.selectbox("Road Surface Type", cat_opts.get("Road_surface_type", ["Asphalt roads", "Earth roads", "Gravel roads"]))
                    light_cond = st.selectbox("Lighting Condition", cat_opts.get("Light_conditions", ["Daylight", "Darkness - lights lit", "Darkness - no lighting"]))
                with cb2:
                    weather_cond = st.selectbox("Weather", cat_opts.get("Weather_conditions", ["Normal", "Raining", "Fog or mist"]))
                    junction_type = st.selectbox("Junction Type", cat_opts.get("Types_of_junction", ["No junction", "Y Shape", "Crossing", "T Shape"]))
            
            with st.expander("💥 Collision Specifics & Casualties", expanded=True):
                cc1, cc2 = st.columns(2)
                with cc1:
                    num_vehicles = st.number_input("Vehicles Involved", min_value=1, max_value=10, value=2, step=1)
                    num_casualties = st.number_input("Total Casualties", min_value=1, max_value=10, value=1, step=1)
                with cc2:
                    time_c = st.time_input("Accident Time", datetime.time(18, 30))
                    day_c = st.selectbox("Day of Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
            
            hour_c = time_c.hour
            if 0 <= hour_c <= 5:
                period_c = "Night"
            elif 6 <= hour_c <= 11:
                period_c = "Morning"
            elif 12 <= hour_c <= 17:
                period_c = "Afternoon"
            else:
                period_c = "Evening"
            
            # Construct feature row with defaults for remaining categorical features
            clf_row = {
                "Number_of_vehicles_involved": num_vehicles,
                "Number_of_casualties": num_casualties,
                "Hour": hour_c,
                "Traffic_Period": period_c,
                "Day_of_week": day_c,
                "Age_band_of_driver": age_band,
                "Sex_of_driver": sex_driver,
                "Educational_level": edu_level,
                "Driving_experience": driving_exp,
                "Type_of_vehicle": vehicle_type,
                "Defect_of_vehicle": vehicle_defect,
                "Road_surface_type": road_surface,
                "Light_conditions": light_cond,
                "Weather_conditions": weather_cond,
                "Types_of_junction": junction_type
            }
            # Fill other cat cols with mode from options
            for col in clf_meta["cat_cols"]:
                if col not in clf_row:
                    clf_row[col] = cat_opts[col][0] if col in cat_opts and cat_opts[col] else "Unknown"
            
            clf_input_df = pd.DataFrame([clf_row])
        
        with col_c_out:
            st.markdown("#### 🎯 Classification Outcome")
            predicted_class = clf_pipe.predict(clf_input_df)[0]
            probabilities = clf_pipe.predict_proba(clf_input_df)[0]
            classes = clf_pipe.classes_
            
            # Class-specific display badge
            if "Fatal" in str(predicted_class):
                st.error(f"### ⚠️ Predicted: **Fatal Injury**")
            elif "Serious" in str(predicted_class):
                st.warning(f"### ⚠️ Predicted: **Serious Injury**")
            else:
                st.success(f"### 🟢 Predicted: **Slight Injury**")
            
            st.markdown("##### 📊 Class Probability Distribution:")
            prob_df = pd.DataFrame({
                "Severity Class": classes,
                "Probability": probabilities
            }).sort_values("Probability", ascending=False)
            
            st.bar_chart(prob_df.set_index("Severity Class"))
            
            for _, r in prob_df.iterrows():
                st.write(f"- **{r['Severity Class']}:** `{r['Probability']*100:.1f}%`")
            
            st.markdown("---")
            st.markdown("##### ⚖️ Rubric & Imbalance Insight:")
            st.info("""
            **Dataset Imbalance Reality:**  
            Slight injuries account for **84.6%** of all records, Serious for **14.2%**, and Fatal for only **1.3%**.  
            Our winning **Bagging Classifier (Weighted F1 = 0.796)** was chosen over raw accuracy because standard models tend to default to the majority class.
            """)

# =============================================================================
# TAB 3: CLUSTERING TRACK
# =============================================================================
with tab_cluster:
    st.markdown("### 🗺️ GPS Trajectory Clustering & Spatial Segmentation")
    st.markdown("Unsupervised partitioning of GPS coordinate logs (Microsoft GeoLife dataset) into **5 natural activity hubs** using **K-Means** ($k=5$, Silhouette $= 0.848$).")
    
    if cluster_bundle is None:
        st.error("⚠️ Clustering model bundle not found in `models/`. Please run `python export_models.py` first.")
    else:
        k_model = cluster_bundle["model"]
        k_scaler = cluster_bundle["scaler"]
        k_meta = cluster_bundle["metadata"]
        
        col_map_in, col_map_out = st.columns([1, 1], gap="large")
        
        with col_map_in:
            st.markdown("#### 📍 Coordinate Explorer")
            
            preset = st.selectbox(
                "Choose a Representative Landmark or Enter Custom Coordinates:",
                [
                    "Haidian District (Tech Hub & University Belt)",
                    "Central Beijing Urban Core (Commercial / Government)",
                    "Beijing Capital International Airport (PEK)",
                    "South Beijing Transit Hub (Daxing Corridor)",
                    "Olympic Green & Chaoyang Residential Belt",
                    "Custom Coordinates..."
                ]
            )
            
            preset_coords = {
                "Haidian District (Tech Hub & University Belt)": (39.982, 116.312),
                "Central Beijing Urban Core (Commercial / Government)": (39.904, 116.407),
                "Beijing Capital International Airport (PEK)": (40.079, 116.603),
                "South Beijing Transit Hub (Daxing Corridor)": (39.752, 116.351),
                "Olympic Green & Chaoyang Residential Belt": (40.015, 116.388)
            }
            
            if preset != "Custom Coordinates...":
                def_lat, def_lon = preset_coords[preset]
            else:
                def_lat, def_lon = 39.95, 116.35
            
            input_lat = st.number_input("Latitude (°N)", min_value=25.0, max_value=45.0, value=float(def_lat), format="%.4f")
            input_lon = st.number_input("Longitude (°E)", min_value=100.0, max_value=125.0, value=float(def_lon), format="%.4f")
            
            # Predict cluster
            scaled_coords = k_scaler.transform([[input_lat, input_lon]])
            assigned_cluster = int(k_model.predict(scaled_coords)[0])
            cluster_name = k_meta["cluster_names"].get(assigned_cluster, f"Cluster {assigned_cluster}")
            
            st.success(f"### 📍 Assigned to: **Cluster {assigned_cluster}**")
            st.markdown(f"**Interpretation:** `{cluster_name}`")
            
            # Distance to centroid
            centroid = k_meta["centers_original"][assigned_cluster]
            dist_km = np.sqrt((input_lat - centroid[0])**2 + (input_lon - centroid[1])**2) * 111.0
            st.metric("Estimated Distance to Cluster Centroid", f"{dist_km:.2f} km")
            
            st.markdown("##### 📐 Internal Clustering Metrics:")
            col_m1, col_m2, col_m3 = st.columns(3)
            col_m1.metric("Silhouette", "0.848", "Excellent separation")
            col_m2.metric("Davies-Bouldin", "0.266", "Lower is better")
            col_m3.metric("Calinski-Harabasz", "260,563", "High density")
            
        with col_map_out:
            st.markdown("#### 🗺️ Spatial Hub Map View")
            
            # Build sample map visualization
            sample_pts = np.array(k_meta["sample_points"])
            sample_df = pd.DataFrame(sample_pts, columns=["lat", "lon"])
            
            # Add user point
            user_point = pd.DataFrame([{"lat": input_lat, "lon": input_lon}])
            
            st.map(sample_df, zoom=9)
            st.caption("Displaying GPS point density across the Beijing metropolitan trajectory sample.")
            
            st.markdown("##### 🥊 K-Means vs Agglomerative Duel:")
            duel_df = pd.DataFrame({
                "Algorithm": ["K-Means (k=5)", "Agglomerative Hierarchical (Average Linkage, k=5)"],
                "Silhouette Score (↑)": [0.8392, 0.8192],
                "Davies-Bouldin Index (↓)": [0.2798, 0.2664],
                "Calinski-Harabasz Index (↑)": [48394.8, 42582.5]
            })
            st.table(duel_df)

# =============================================================================
# TAB 4: SCOREBOARD & FULL LEADERBOARD
# =============================================================================
with tab_score:
    st.markdown("### 🏆 Consolidated Capstone Scoreboard (22 Models Evaluated)")
    st.markdown("Comprehensive evaluation across all three tracks following strict identical splits and leakage-free cross validation.")
    
    track_filter = st.radio("Select Track View:", ["All Tracks", "Regression (10 Models)", "Classification (10 Models)", "Clustering (2 Models)"], horizontal=True)
    
    if track_filter in ["All Tracks", "Regression (10 Models)"]:
        st.markdown("#### 📈 Track 1: Regression Leaderboard (Held-out Test Split)")
        reg_table = pd.DataFrame([
            {"Rank": 1, "Algorithm": "Random Forest (Tuned)", "R² Score": 0.9561, "RMSE": 418.7, "MAE": 230.1, "Notes": "Winner · 84% variance from hour & period"},
            {"Rank": 2, "Algorithm": "Gradient Boosting (Tuned)", "R² Score": 0.9524, "RMSE": 436.0, "MAE": 259.1, "Notes": "Second best · sequential error correction"},
            {"Rank": 3, "Algorithm": "Decision Tree", "R² Score": 0.9417, "RMSE": 482.9, "MAE": 283.2, "Notes": "Strong baseline tree split"},
            {"Rank": 4, "Algorithm": "Polynomial Regression (deg 2)", "R² Score": 0.8510, "RMSE": 771.8, "MAE": 575.6, "Notes": "2,849 expanded interaction features"},
            {"Rank": 5, "Algorithm": "K-Nearest Neighbors", "R² Score": 0.8306, "RMSE": 822.9, "MAE": 579.3, "Notes": "Sensitive to distance scaling"},
            {"Rank": 6, "Algorithm": "Support Vector Regressor (SVR)", "R² Score": 0.8294, "RMSE": 826.0, "MAE": 601.2, "Notes": "Radial Basis Function kernel"},
            {"Rank": 7, "Algorithm": "Lasso Regression", "R² Score": 0.7314, "RMSE": 1036.3, "MAE": 795.3, "Notes": "L1 penalty for feature sparsity"},
            {"Rank": 8, "Algorithm": "Ridge Regression", "R² Score": 0.7312, "RMSE": 1036.6, "MAE": 795.5, "Notes": "L2 penalty shrinks coefficients"},
            {"Rank": 9, "Algorithm": "Linear Regression", "R² Score": 0.7311, "RMSE": 1036.8, "MAE": 795.6, "Notes": "Baseline linear sum"},
            {"Rank": 10, "Algorithm": "ElasticNet Regression", "R² Score": 0.7169, "RMSE": 1063.8, "MAE": 842.8, "Notes": "Combined L1 + L2 penalty"}
        ])
        st.dataframe(reg_table, use_container_width=True, hide_index=True)
    
    if track_filter in ["All Tracks", "Classification (10 Models)"]:
        st.markdown("#### 🚨 Track 2: Classification Leaderboard (Consolidated 10-Model Table)")
        clf_table = pd.DataFrame([
            {"Rank": 1, "Algorithm": "Bagging Classifier (Decision Tree)", "Accuracy": "85.23%", "Precision (w)": 0.8360, "Recall (w)": 0.8523, "F1-Score (w)": 0.7957, "ROC-AUC (OvR)": 0.7264, "Status": "Selected Best Model"},
            {"Rank": 2, "Algorithm": "Random Forest (Tuned)", "Accuracy": "84.66%", "Precision (w)": 0.8576, "Recall (w)": 0.8466, "F1-Score (w)": 0.7940, "ROC-AUC (OvR)": 0.6858, "Status": "Improved +0.019 via CV"},
            {"Rank": 3, "Algorithm": "K-Nearest Neighbors", "Accuracy": "83.85%", "Precision (w)": 0.7784, "Recall (w)": 0.8385, "F1-Score (w)": 0.7893, "ROC-AUC (OvR)": 0.6181, "Status": "Part A Runner Up"},
            {"Rank": 4, "Algorithm": "Gradient Boosting", "Accuracy": "84.70%", "Precision (w)": 0.8054, "Recall (w)": 0.8470, "F1-Score (w)": 0.7868, "ROC-AUC (OvR)": 0.6501, "Status": "Ensemble"},
            {"Rank": 5, "Algorithm": "AdaBoost Classifier", "Accuracy": "84.58%", "Precision (w)": 0.7153, "Recall (w)": 0.8458, "F1-Score (w)": 0.7751, "ROC-AUC (OvR)": 0.6145, "Status": "Ensemble"},
            {"Rank": 6, "Algorithm": "MLP Classifier (Neural Net)", "Accuracy": "77.76%", "Precision (w)": 0.7621, "Recall (w)": 0.7776, "F1-Score (w)": 0.7695, "ROC-AUC (OvR)": 0.6034, "Status": "Best minority class recall"},
            {"Rank": 7, "Algorithm": "Decision Tree Classifier", "Accuracy": "75.93%", "Precision (w)": 0.7676, "Recall (w)": 0.7593, "F1-Score (w)": 0.7633, "ROC-AUC (OvR)": 0.5611, "Status": "Single tree split"},
            {"Rank": 8, "Algorithm": "Support Vector Machine (SVC)", "Accuracy": "69.24%", "Precision (w)": 0.7850, "Recall (w)": 0.6924, "F1-Score (w)": 0.7267, "ROC-AUC (OvR)": "—", "Status": "Margin boundary"},
            {"Rank": 9, "Algorithm": "Logistic Regression", "Accuracy": "50.65%", "Precision (w)": 0.7611, "Recall (w)": 0.5065, "F1-Score (w)": 0.5902, "ROC-AUC (OvR)": "—", "Status": "Baseline classifier"},
            {"Rank": 10, "Algorithm": "Gaussian Naive Bayes", "Accuracy": "9.13%", "Precision (w)": 0.7712, "Recall (w)": 0.0913, "F1-Score (w)": 0.1241, "ROC-AUC (OvR)": "—", "Status": "Violated independence assumption"}
        ])
        st.dataframe(clf_table, use_container_width=True, hide_index=True)

    if track_filter in ["All Tracks", "Clustering (2 Models)"]:
        st.markdown("#### 🗺️ Track 3: Clustering Comparison (Identical 2,000 Points)")
        clust_table = pd.DataFrame([
            {"Algorithm": "K-Means (k=5)", "Silhouette Score": "0.8392", "Davies-Bouldin": "0.2798", "Calinski-Harabasz": "48,394.8", "Winner Metric": "Silhouette & Calinski-Harabasz"},
            {"Algorithm": "Agglomerative Hierarchical (Average)", "Silhouette Score": "0.8192", "Davies-Bouldin": "0.2664", "Calinski-Harabasz": "42,582.5", "Winner Metric": "Davies-Bouldin Index"}
        ])
        st.dataframe(clust_table, use_container_width=True, hide_index=True)

# =============================================================================
# TAB 5: ABOUT & VIVA PREPARATION
# =============================================================================
with tab_about:
    st.markdown("### ℹ️ Project Architecture & Engineering Methodology")
    
    st.markdown("""
    #### 🎯 Core Problem Statement
    *Roads generate three fundamental questions, and each requires a distinct machine learning approach:*
    1. **Regression:** *"How many vehicles will travel on the interstate at a given hour under current weather?"*  
       $\rightarrow$ Solved with **Tuned Random Forest ($R^2 = 0.956$)**.
    2. **Classification:** *"Given an accident scenario, how severe will the injury outcome be?"*  
       $\rightarrow$ Solved with **Bagging Classifier (F1 = 0.796)** overcoming extreme class imbalance.
    3. **Clustering:** *"Where do vehicle GPS points naturally concentrate into functional spatial clusters?"*  
       $\rightarrow$ Solved with **K-Means ($k=5$, Silhouette $= 0.848$)**.
    
    #### 🔒 Three Rules We Never Broke (Rubric Section 7)
    - **Zero Data Leakage:** All imputers, standardizers, and one-hot encoders were fitted **strictly on the training split** and transformed onto the test split.
    - **Fair Comparison:** Within each track, every model was evaluated on the **exact same train/test split (80/20)** with fixed seed `42`.
    - **Evidence-Based Model Selection:** 5-fold cross validation was conducted for the top-performing models before tuning.
    """)
    
    st.markdown("---")
    st.markdown("### 👥 Engineering Team & Track Ownership")
    c_m1, c_m2, c_m3 = st.columns(3)
    with c_m1:
        st.info("""
        **K. Sai charan**  
        *Track 1: Regression Lead*  
        - Metro Interstate Volume Pipeline
        - 10 Regression Algorithms comparison
        - Residual analysis & feature importance
        """)
    with c_m2:
        st.info("""
        **L. Hrudhay**  
        *Track 2: Classification Lead*  
        - RTA Severity Classification Pipeline
        - Class imbalance analysis & metrics
        - Part A & Part B consolidated duel
        """)
    with c_m3:
        st.info("""
        **K. Sapthagiri Reddy**  
        *Track 3: Clustering Lead*  
        - GeoLife GPS sampling & scaling
        - K-Means Elbow method & Dendrogram
        - PCA & t-SNE 2D cluster visualization
        """)
