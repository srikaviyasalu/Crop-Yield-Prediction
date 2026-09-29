import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==========================================
# STREAMLIT CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Crop Yield Prediction",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling Tab Navigation Contrast
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #2E7D32;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #555;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: 800;
        color: #1976D2;
    }
    .metric-label {
        font-size: 1.2rem;
        color: #666;
        font-weight: 600;
    }
    
    /* Compact Dashboard Style Navigation */
    .stTabs [data-baseweb="tab-list"] {
        gap: 32px;
        border-bottom: 1px solid #444;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 0px;
        background-color: transparent !important;
        border-radius: 0;
        border: none;
    }
    .stTabs [data-baseweb="tab"] p {
        color: #b0b0b0;
        font-weight: 600;
        font-size: 1.1rem;
    }
    .stTabs [aria-selected="true"] {
        background-color: transparent !important;
        border-bottom: 3px solid #4CAF50;
    }
    .stTabs [aria-selected="true"] p {
        color: #ffffff;
    }
    .stTabs [data-baseweb="tab"]:hover p {
        color: #ffffff;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: transparent !important;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# DATA LOADING & PREPROCESSING
# ==========================================
@st.cache_data
def load_data():
    file_path = os.path.join("data", "crop_yield.csv")
    if not os.path.exists(file_path):
        st.error(f"Dataset not found at {file_path}. Please ensure the file exists.")
        st.stop()
    
    df = pd.read_csv(file_path)
    
    expected_columns = ['Crop', 'Crop_Year', 'Season', 'State', 'Area', 'Production', 'Annual_Rainfall', 'Fertilizer', 'Pesticide', 'Yield']
    missing_cols = [col for col in expected_columns if col not in df.columns]
    if missing_cols:
        st.error(f"Missing required columns in dataset: {', '.join(missing_cols)}")
        st.stop()
        
    return df

@st.cache_resource
def train_model(df):
    """
    Trains the Random Forest Regression model.
    Crop_Year is EXCLUDED to prevent invalid time-series extrapolation.
    Production is EXCLUDED to prevent target leakage.
    Yield is the target variable.
    """
    target = 'Yield'
    
    # Exclude 'Production', 'Crop_Year', and 'Yield' from features
    feature_cols = ['Crop', 'Season', 'State', 'Area', 'Annual_Rainfall', 'Fertilizer', 'Pesticide']
    
    X = df[feature_cols]
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    
    categorical_features = ['Crop', 'Season', 'State']
    numerical_features = ['Area', 'Annual_Rainfall', 'Fertilizer', 'Pesticide']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ])
    
    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    metrics = {
        'mae': mean_absolute_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'r2': r2_score(y_test, y_pred),
        'y_test': y_test,
        'y_pred': y_pred
    }
    
    return pipeline, metrics, feature_cols, numerical_features, categorical_features

# ==========================================
# MAIN APPLICATION
# ==========================================
def main():
    st.markdown('<p class="main-header">🌾 Crop Yield Prediction</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Estimate crop yield using historical agricultural data.</p>', unsafe_allow_html=True)
    
    with st.spinner("Loading agricultural data..."):
        df = load_data()
        
    with st.spinner("Initializing decision support engine..."):
        pipeline, metrics, feature_cols, num_cols, cat_cols = train_model(df)
        
    # Define historical ranges for validation
    ranges = {
        'Area': (df['Area'].min(), df['Area'].max()),
        'Annual_Rainfall': (df['Annual_Rainfall'].min(), df['Annual_Rainfall'].max()),
        'Fertilizer': (df['Fertilizer'].min(), df['Fertilizer'].max()),
        'Pesticide': (df['Pesticide'].min(), df['Pesticide'].max())
    }
    
    # Get a REAL representative row for realistic UI defaults
    # Find the most frequent Crop, State, Season combination in the dataset
    common_combo = df.groupby(['Crop', 'State', 'Season']).size().idxmax()
    combo_df = df[(df['Crop'] == common_combo[0]) & (df['State'] == common_combo[1]) & (df['Season'] == common_combo[2])]
    
    # Select exactly one actual row to guarantee all default values existed together historically
    real_row = combo_df.iloc[0]
    def_crop = real_row['Crop']
    def_state = real_row['State']
    def_season = real_row['Season']
    def_area = float(real_row['Area'])
    def_rain = float(real_row['Annual_Rainfall'])
    def_fert = float(real_row['Fertilizer'])
    def_pest = float(real_row['Pesticide'])
        
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔮 Predict Yield", 
        "🔄 What-If Analysis", 
        "🌾 Compare Crops", 
        "📈 Crop Trends", 
        "💡 Insights"
    ])
    
    # ----------------------------------------------------
    # TAB 1: PREDICT YIELD
    # ----------------------------------------------------
    with tab1:
        st.write("Enter agricultural inputs below to estimate crop yield.")
        st.info("Estimated 2026 Yield is based on historical agricultural data available through 2020 and the conditions you entered.")
        
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Agricultural Parameters**")
                crop = st.selectbox("Crop", sorted(df['Crop'].unique()), index=list(sorted(df['Crop'].unique())).index(def_crop))
                season = st.selectbox("Season", sorted(df['Season'].unique()), index=list(sorted(df['Season'].unique())).index(def_season))
                state = st.selectbox("State", sorted(df['State'].unique()), index=list(sorted(df['State'].unique())).index(def_state))
                # 2026 UX - only used for labeling, not in model prediction
                display_year = st.number_input("Year", min_value=1990, max_value=2050, value=2026, step=1)
                
            with col2:
                st.markdown("**Environmental & Input Parameters**")
                area = st.number_input("Area (Historical dataset value)", min_value=0.0, value=def_area, format="%.2f")
                rainfall = st.number_input("Annual Rainfall (Historical rainfall value)", min_value=0.0, value=def_rain, format="%.2f")
                fertilizer = st.number_input("Fertilizer (Dataset value)", min_value=0.0, value=def_fert, format="%.2f")
                pesticide = st.number_input("Pesticide (Dataset value)", min_value=0.0, value=def_pest, format="%.2f")
                
            submit_button = st.form_submit_button("PREDICT YIELD", use_container_width=True)
            
        if submit_button:
            warnings = []
            if not (ranges['Area'][0] <= area <= ranges['Area'][1]): warnings.append("Area")
            if not (ranges['Annual_Rainfall'][0] <= rainfall <= ranges['Annual_Rainfall'][1]): warnings.append("Annual Rainfall")
            if not (ranges['Fertilizer'][0] <= fertilizer <= ranges['Fertilizer'][1]): warnings.append("Fertilizer")
            if not (ranges['Pesticide'][0] <= pesticide <= ranges['Pesticide'][1]): warnings.append("Pesticide")
            
            if warnings:
                st.warning(f"⚠️ The following values are outside the historical range used by the model: **{', '.join(warnings)}**. The estimate may be less reliable.")
                
            input_data = pd.DataFrame([{
                'Crop': crop, 'Season': season, 'State': state,
                'Area': area, 'Annual_Rainfall': rainfall, 'Fertilizer': fertilizer, 'Pesticide': pesticide
            }])
            
            prediction = pipeline.predict(input_data)[0]
            
            year_label = f"Estimated {display_year} Yield" if display_year > df['Crop_Year'].max() else f"Estimated Yield for {display_year}"
            
            st.markdown(f"""
            <div class="metric-card" style="background-color: #E8F5E9; border-left: 5px solid #4CAF50; margin-top: 20px;">
                <div class="metric-label">{year_label}</div>
                <div class="metric-value" style="color: #2E7D32;">{prediction:.2f}</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("### 🌱 Prediction Conditions")
            
            # Compact 2-column layout for the conditions summary
            sum_col1, sum_col2 = st.columns(2)
            with sum_col1:
                st.markdown(f"**Crop:** {crop}")
                st.markdown(f"**Season:** {season}")
                st.markdown(f"**State:** {state}")
                st.markdown(f"**Prediction Year:** {display_year}")
            with sum_col2:
                st.markdown(f"**Area:** {area:.2f}")
                st.markdown(f"**Annual Rainfall:** {rainfall:.2f}")
                st.markdown(f"**Fertilizer:** {fertilizer:.2f}")
                st.markdown(f"**Pesticide:** {pesticide:.2f}")
                
            st.info("💡 **About this estimate:** This estimate is based on historical agricultural data available through 2020 and the conditions you entered. Actual estimated yield may differ because of weather, soil conditions, farming practices, and other factors not represented in the dataset.")
            
    # ----------------------------------------------------
    # TAB 2: WHAT-IF ANALYSIS
    # ----------------------------------------------------
    with tab2:
        st.write("Establish a baseline scenario and adjust parameters to see how changes affect your estimated crop yield.")
        
        st.markdown("### 1. Set Baseline Variables")
        col_b1, col_b2, col_b3, col_b4 = st.columns(4)
        with col_b1:
            w_crop = st.selectbox("Crop (Base)", sorted(df['Crop'].unique()), index=list(sorted(df['Crop'].unique())).index(def_crop), key="w_crop")
            w_year = st.number_input("Year (Base)", value=2026, key="w_year")
        with col_b2:
            w_season = st.selectbox("Season (Base)", sorted(df['Season'].unique()), index=list(sorted(df['Season'].unique())).index(def_season), key="w_season")
            w_state = st.selectbox("State (Base)", sorted(df['State'].unique()), index=list(sorted(df['State'].unique())).index(def_state), key="w_state")
        with col_b3:
            w_area = st.number_input("Area (Historical dataset value)", value=def_area, key="w_area")
            w_rain = st.number_input("Rainfall (Historical rainfall value)", value=def_rain, key="w_rain")
        with col_b4:
            w_fert = st.number_input("Fertilizer (Dataset value)", value=def_fert, key="w_fert")
            w_pest = st.number_input("Pesticide (Dataset value)", value=def_pest, key="w_pest")
            
        st.markdown("### 2. Modify Scenario Conditions")
        st.write("Adjust multipliers to simulate changes (e.g., 1.20 = 20% increase).")
        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        with col_s1:
            s_area_factor = st.slider("Area Multiplier", 0.5, 2.0, 1.0, 0.1)
        with col_s2:
            s_rain_factor = st.slider("Rainfall Multiplier", 0.5, 2.0, 1.0, 0.1)
        with col_s3:
            s_fert_factor = st.slider("Fertilizer Multiplier", 0.5, 2.0, 1.0, 0.1)
        with col_s4:
            s_pest_factor = st.slider("Pesticide Multiplier", 0.5, 2.0, 1.0, 0.1)
            
        if st.button("RUN SCENARIO ANALYSIS", type="primary"):
            base_data = pd.DataFrame([{
                'Crop': w_crop, 'Season': w_season, 'State': w_state,
                'Area': w_area, 'Annual_Rainfall': w_rain, 'Fertilizer': w_fert, 'Pesticide': w_pest
            }])
            
            scen_area = w_area * s_area_factor
            scen_rain = w_rain * s_rain_factor
            scen_fert = w_fert * s_fert_factor
            scen_pest = w_pest * s_pest_factor
            
            scenario_data = pd.DataFrame([{
                'Crop': w_crop, 'Season': w_season, 'State': w_state,
                'Area': scen_area, 'Annual_Rainfall': scen_rain, 'Fertilizer': scen_fert, 'Pesticide': scen_pest
            }])
            
            base_pred = pipeline.predict(base_data)[0]
            scenario_pred = pipeline.predict(scenario_data)[0]
            difference = scenario_pred - base_pred
            pct_change = (difference / base_pred) * 100 if base_pred > 0 else 0
            
            st.markdown("### Scenario Comparison")
            
            # Show actual applied values
            comp_df = pd.DataFrame({
                "Parameter": ["Area", "Annual Rainfall", "Fertilizer", "Pesticide"],
                "Baseline Value": [f"{w_area:.2f}", f"{w_rain:.2f}", f"{w_fert:.2f}", f"{w_pest:.2f}"],
                "Scenario Value": [f"{scen_area:.2f}", f"{scen_rain:.2f}", f"{scen_fert:.2f}", f"{scen_pest:.2f}"]
            })
            st.table(comp_df)
            
            res_col1, res_col2, res_col3 = st.columns(3)
            with res_col1:
                st.metric("Baseline Estimated Yield", f"{base_pred:.2f}")
            with res_col2:
                st.metric("Scenario Estimated Yield", f"{scenario_pred:.2f}", delta=f"{pct_change:.1f}%")
            with res_col3:
                st.metric("Absolute Change", f"{difference:+.2f}")
                
            fig = go.Figure(data=[
                go.Bar(name='Baseline', x=['Estimated Yield'], y=[base_pred], marker_color='#1976D2'),
                go.Bar(name='Scenario', x=['Estimated Yield'], y=[scenario_pred], marker_color='#4CAF50')
            ])
            fig.update_layout(barmode='group', title="Yield Comparison", height=400, width=500)
            st.plotly_chart(fig, use_container_width=True)

    # ----------------------------------------------------
    # TAB 3: COMPARE CROPS
    # ----------------------------------------------------
    with tab3:
        st.write("Compare estimated yields for multiple crops under the exact same environmental and agricultural conditions.")
        
        c_col1, c_col2 = st.columns([1, 2])
        with c_col1:
            st.markdown("### 1. Set Conditions")
            c_year = st.number_input("Year", value=2026, key="c_year")
            c_season = st.selectbox("Season", sorted(df['Season'].unique()), index=list(sorted(df['Season'].unique())).index(def_season), key="c_season")
            c_state = st.selectbox("State", sorted(df['State'].unique()), index=list(sorted(df['State'].unique())).index(def_state), key="c_state")
            c_area = st.number_input("Area (Historical dataset value)", value=def_area, key="c_area")
            c_rain = st.number_input("Rainfall (Historical rainfall value)", value=def_rain, key="c_rain")
            c_fert = st.number_input("Fertilizer (Dataset value)", value=def_fert, key="c_fert")
            c_pest = st.number_input("Pesticide (Dataset value)", value=def_pest, key="c_pest")
            
        with c_col2:
            st.markdown("### 2. Select Crops to Compare")
            selected_crops = st.multiselect("Choose two or more crops", sorted(df['Crop'].unique()), default=[def_crop])
            
            if st.button("COMPARE CROPS", type="primary"):
                if len(selected_crops) < 2:
                    st.warning("Please select at least two crops to compare.")
                else:
                    results = []
                    for cr in selected_crops:
                        input_data = pd.DataFrame([{
                            'Crop': cr, 'Season': c_season, 'State': c_state,
                            'Area': c_area, 'Annual_Rainfall': c_rain, 'Fertilizer': c_fert, 'Pesticide': c_pest
                        }])
                        pred = pipeline.predict(input_data)[0]
                        results.append({"Crop": cr, "Estimated Yield": pred})
                        
                    res_df = pd.DataFrame(results).sort_values(by="Estimated Yield", ascending=False)
                    
                    st.markdown("#### Comparison Results")
                    st.info("These are model estimates under the same entered conditions. They are not guaranteed outcomes or crop recommendations.")
                    
                    fig = px.bar(res_df, x='Crop', y='Estimated Yield', color='Crop', 
                                 title=f"Estimated Yield Comparison for {c_year}",
                                 text_auto='.2f')
                    fig.update_layout(showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.dataframe(res_df, use_container_width=True, hide_index=True)

    # ----------------------------------------------------
    # TAB 4: CROP TRENDS
    # ----------------------------------------------------
    with tab4:
        st.write("View historical yield trends for specific crops and regions based on actual recorded data.")
        
        t_col1, t_col2, t_col3 = st.columns(3)
        with t_col1:
            t_crop = st.selectbox("Select Crop", sorted(df['Crop'].unique()), index=list(sorted(df['Crop'].unique())).index(def_crop), key="t_crop")
        with t_col2:
            t_state = st.selectbox("Select State", sorted(df['State'].unique()), index=list(sorted(df['State'].unique())).index(def_state), key="t_state")
        with t_col3:
            t_season = st.selectbox("Select Season", sorted(df['Season'].unique()), index=list(sorted(df['Season'].unique())).index(def_season), key="t_season")
            
        historical_data = df[(df['Crop'] == t_crop) & (df['State'] == t_state) & (df['Season'] == t_season)]
        
        if len(historical_data) < 2:
            st.info(f"Insufficient historical records found for {t_crop} in {t_state} during {t_season} season. Please try a different combination.")
        else:
            # Group by year safely
            trend_df = historical_data.groupby('Crop_Year', as_index=False)['Yield'].mean()
            trend_df = trend_df.sort_values('Crop_Year')
            
            formatted_season = t_season.strip()
            
            fig = px.line(trend_df, x='Crop_Year', y='Yield', markers=True,
                          title=f"Historical Actual Yield for {t_crop} in {t_state} ({formatted_season})",
                          labels={"Crop_Year": "Historical Year", "Yield": "Actual Yield"})
            fig.update_layout(xaxis=dict(tickmode='linear', dtick=1))
            st.plotly_chart(fig, use_container_width=True)
            
            st.write("Note: This chart displays only actual historical records available in the dataset. It does not display estimated future data.")

    # ----------------------------------------------------
    # TAB 5: INSIGHTS
    # ----------------------------------------------------
    with tab5:
        st.write("Technical insights into the crop yield estimation model and historical dataset.")
        
        insight_type = st.radio("Select Insight View:", ["Model Performance", "Explainability", "Dataset Overview"], horizontal=True)
        st.divider()
        
        if insight_type == "Model Performance":
            st.subheader("Historical Test Set Performance")
            st.write("These metrics evaluate how well the model learned from the historical dataset, tested on a random 20% unseen data split.")
            
            m_col1, m_col2, m_col3 = st.columns(3)
            with m_col1:
                st.markdown(f'<div class="metric-card"><div class="metric-label">Mean Absolute Error (MAE)</div><div class="metric-value" style="font-size:2rem;">{metrics["mae"]:.2f}</div></div>', unsafe_allow_html=True)
            with m_col2:
                st.markdown(f'<div class="metric-card"><div class="metric-label">Root Mean Squared Error (RMSE)</div><div class="metric-value" style="font-size:2rem;">{metrics["rmse"]:.2f}</div></div>', unsafe_allow_html=True)
            with m_col3:
                st.markdown(f'<div class="metric-card"><div class="metric-label">R² Score</div><div class="metric-value" style="font-size:2rem;">{metrics["r2"]:.4f}</div><div style="font-size:0.8rem; color:#666; margin-top:5px;">R² measures how well the model explains variation in the target. It is not classification accuracy.</div></div>', unsafe_allow_html=True)
                
            st.markdown("<br>", unsafe_allow_html=True)
            st.subheader("Actual vs Predicted Yield Visualization")
            st.write("A scatter plot showing a reproducible random sample of 500 historical test predictions versus their true values. The dashed line represents perfect prediction (1:1).")
            
            results_df = pd.DataFrame({'Actual': metrics['y_test'], 'Predicted': metrics['y_pred']})
            
            tab_full, tab_detail = st.tabs(["Full View", "Detailed View (Central 95%)"])
            
            with tab_full:
                sample_size = min(500, len(results_df))
                viz_df = results_df.sample(sample_size, random_state=42)
                
                fig = px.scatter(viz_df, x='Actual', y='Predicted', 
                                 opacity=0.6,
                                 labels={'Actual': 'Actual Historical Yield', 'Predicted': 'Model Predicted Yield'},
                                 color_discrete_sequence=['#FF9800'])
                
                min_val = min(viz_df['Actual'].min(), viz_df['Predicted'].min())
                max_val = max(viz_df['Actual'].max(), viz_df['Predicted'].max())
                fig.add_shape(type='line', line=dict(dash='dash', color='gray'),
                              x0=min_val, y0=min_val, x1=max_val, y1=max_val)
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
                
            with tab_detail:
                st.write("Detailed view for observations within the central 95% of actual-yield values. Evaluation metrics above use the complete test set.")
                
                # Filter out extreme top 5% actual values for clearer plotting
                percentile_95 = results_df['Actual'].quantile(0.95)
                detailed_results = results_df[results_df['Actual'] <= percentile_95]
                
                sample_size_det = min(500, len(detailed_results))
                viz_df_det = detailed_results.sample(sample_size_det, random_state=42)
                
                fig_det = px.scatter(viz_df_det, x='Actual', y='Predicted', 
                                 opacity=0.6,
                                 labels={'Actual': 'Actual Historical Yield', 'Predicted': 'Model Predicted Yield'},
                                 color_discrete_sequence=['#4CAF50'])
                
                min_val_det = min(viz_df_det['Actual'].min(), viz_df_det['Predicted'].min())
                max_val_det = max(viz_df_det['Actual'].max(), viz_df_det['Predicted'].max())
                fig_det.add_shape(type='line', line=dict(dash='dash', color='gray'),
                              x0=min_val_det, y0=min_val_det, x1=max_val_det, y1=max_val_det)
                fig_det.update_layout(height=500)
                st.plotly_chart(fig_det, use_container_width=True)
            
        elif insight_type == "Explainability":
            st.subheader("Feature Importance")
            st.info("Which features were most important to the Random Forest model's predictions. The model relied strongly on these features when making predictions. This does not imply causation.")
            
            model = pipeline.named_steps['model']
            preprocessor = pipeline.named_steps['preprocessor']
            cat_encoder = preprocessor.named_transformers_['cat']
            cat_feature_names = cat_encoder.get_feature_names_out(cat_cols)
            
            all_feature_names = num_cols + list(cat_feature_names)
            importances = model.feature_importances_
            
            feature_importance_map = {feat: 0.0 for feat in num_cols + cat_cols}
            
            for name, imp in zip(all_feature_names, importances):
                if name in num_cols:
                    feature_importance_map[name] += imp
                else:
                    for cat_col in cat_cols:
                        if name.startswith(cat_col + "_"):
                            feature_importance_map[cat_col] += imp
                            break
                            
            fi_df = pd.DataFrame({
                'Feature': list(feature_importance_map.keys()),
                'Importance': list(feature_importance_map.values())
            })
            
            fi_df['Importance'] = (fi_df['Importance'] / fi_df['Importance'].sum()) * 100
            fi_df = fi_df.sort_values(by='Importance', ascending=True)
            
            fig = px.bar(fi_df, x='Importance', y='Feature', orientation='h',
                         labels={'Importance': 'Relative Importance (%)', 'Feature': 'Agricultural Feature'},
                         color='Importance',
                         color_continuous_scale='Greens')
            
            st.plotly_chart(fig, use_container_width=True)
            
        elif insight_type == "Dataset Overview":
            st.subheader("Historical Dataset Overview")
            
            s_col1, s_col2, s_col3, s_col4 = st.columns(4)
            with s_col1:
                st.metric("Total Historical Records", f"{len(df):,}")
            with s_col2:
                st.metric("Total Features", df.shape[1])
            with s_col3:
                st.metric("Unique Crops", df['Crop'].nunique())
            with s_col4:
                st.metric("Unique States", df['State'].nunique())
                
            st.write(f"**Available Historical Years:** {df['Crop_Year'].min()} to {df['Crop_Year'].max()}")
            st.write(f"**Available Seasons:** {', '.join(df['Season'].unique())}")
            
            st.markdown("### Feature Configuration")
            st.write("**Target Variable:** `Yield`")
            st.write("**Excluded Feature:** `Production`")
            st.warning("""
            **Why is Production excluded?** Production is directly related to crop output and cultivated area. Including it can provide information closely tied to the target Yield, which can introduce target leakage and make the prediction scenario less realistic.
            """)
            
            st.subheader("Data Preview")
            st.dataframe(df.head(50), use_container_width=True)

if __name__ == "__main__":
    main()
