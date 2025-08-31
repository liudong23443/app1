import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.colors as mcolors
import os
import platform
import io
import warnings
import tempfile
import urllib.request
warnings.filterwarnings('ignore')

# Set page configuration - ensure this is the first Streamlit command
st.set_page_config(
    page_title="Gastric Cancer Postoperative Survival Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Try to download and install Source Han Sans font
def download_and_setup_chinese_font():
    try:
        # Create temporary directory for font files
        tmp_dir = tempfile.mkdtemp()
        font_url = "https://github.com/adobe-fonts/source-han-sans/raw/release/OTF/SimplifiedChinese/SourceHanSansSC-Regular.otf"
        font_path = os.path.join(tmp_dir, "SourceHanSansSC-Regular.otf")
        
        # Download font file
        urllib.request.urlretrieve(font_url, font_path)
        
        # Check if file was downloaded successfully
        if os.path.exists(font_path) and os.path.getsize(font_path) > 0:
            # Add font path to matplotlib configuration
            mpl.font_manager.fontManager.addfont(font_path)
            
            # Add font path to matplotlib's font path list
            font_dirs = [tmp_dir]
            font_files = mpl.font_manager.findSystemFonts(fontpaths=font_dirs)
            for font_file in font_files:
                mpl.font_manager.fontManager.addfont(font_file)
            
            # Refresh matplotlib's font cache
            try:
                mpl.font_manager._rebuild()
            except:
                pass  # If rebuild fails, continue using loaded fonts
            
            # Set font configuration
            plt.rcParams['font.sans-serif'] = ['Source Han Sans SC', 'DejaVu Sans', 'Arial']
            plt.rcParams['axes.unicode_minus'] = False
            plt.rcParams['font.family'] = 'sans-serif'
            
            return True, font_path, None
        else:
            return False, None, "Font file download failed"
    except Exception as e:
        return False, None, f"Error downloading font: {str(e)}"

# Try to download and set Chinese font
font_downloaded, font_path, error_message = download_and_setup_chinese_font()

if not font_downloaded:
    # Use common Chinese fonts as alternatives
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 
                                     'Noto Sans CJK JP', 'Noto Sans CJK SC', 'WenQuanYi Micro Hei',
                                     'DejaVu Sans', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.family'] = 'sans-serif'

# Ensure plotly can display Chinese
import plotly.io as pio
pio.templates.default = "simple_white"

# Custom CSS styles
st.markdown("""
<style>
    .main-header {
        font-size: 1.8rem;
        color: white;
        text-align: center;
        margin-bottom: 0.5rem;
        font-family: system-ui, -apple-system, 'Segoe UI', Roboto, 'Microsoft YaHei', 'SimHei', sans-serif;
        padding: 0.8rem 0;
        border-bottom: 2px solid #E5E7EB;
    }
    .sub-header {
        font-size: 1.2rem;
        color: white;
        margin-top: 0.5rem;
        margin-bottom: 0.5rem;
        font-family: system-ui, -apple-system, 'Segoe UI', Roboto, 'Microsoft YaHei', 'SimHei', sans-serif;
    }
    .description {
        font-size: 1rem;
        color: #4B5563;
        margin-bottom: 1rem;
        padding: 0.5rem;
        background-color: #F3F4F6;
        border-radius: 0.5rem;
        border-left: 4px solid #1E3A8A;
    }
    .section-container {
        padding: 0.8rem;
        background-color: #F9FAFB;
        border-radius: 0.5rem;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
        margin-bottom: 0.8rem;
        height: 100%;
    }
    .results-container {
        padding: 0.8rem;
        background-color: #F0F9FF;
        border-radius: 0.5rem;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
        margin-bottom: 0.8rem;
        border: 1px solid #93C5FD;
        height: 100%;
    }
    .metric-card {
        background-color: #F0F9FF;
        padding: 0.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        text-align: center;
    }
    .disclaimer {
        font-size: 0.75rem;
        color: #6B7280;
        text-align: center;
        margin-top: 0.5rem;
        padding-top: 0.5rem;
        border-top: 1px solid #E5E7EB;
    }
    .stButton>button {
        background-color: #1E3A8A;
        color: white;
        font-weight: bold;
        padding: 0.5rem 1rem;
        font-size: 1rem;
        border-radius: 0.3rem;
        border: none;
        margin-top: 0.5rem;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #1E40AF;
    }
    /* Improve responsive layout on small devices */
    @media (max-width: 1200px) {
        .main-header {
            font-size: 1.5rem;
        }
        .sub-header {
            font-size: 1.1rem;
        }
    }
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
    /* Optimize metric display */
    .stMetric {
        background-color: transparent;
        padding: 5px;
        border-radius: 5px;
    }
    /* Improve dividers */
    hr {
        margin: 0.8rem 0;
        border: 0;
        height: 1px;
        background-image: linear-gradient(to right, rgba(0,0,0,0), rgba(0,0,0,0.1), rgba(0,0,0,0));
    }
    /* Darken text in dashboards and SHAP plots */
    .js-plotly-plot .plotly .gtitle {
        font-weight: bold !important;
        fill: #000000 !important;
    }
    .js-plotly-plot .plotly .g-gtitle {
        font-weight: bold !important;
        fill: #000000 !important;
    }
    /* Chart background */
    .stPlotlyChart, .stImage {
        background-color: white !important;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.1rem !important;
        font-weight: bold !important;
        color: #1E3A8A !important;
    }
    div[data-testid="stMetricLabel"] {
        font-weight: bold !important;
        font-size: 0.9rem !important;
    }
    /* Compact sliders and radio buttons */
    div.row-widget.stRadio > div {
        flex-direction: row;
        align-items: center;
    }
    div.row-widget.stRadio > div[role="radiogroup"] > label {
        padding: 0.2rem 0.5rem;
        min-height: auto;
    }
    div.stSlider {
        padding-top: 0.3rem;
        padding-bottom: 0.5rem;
    }
    /* Compact labels */
    p {
        margin-bottom: 0.3rem;
    }
    div.stMarkdown p {
        margin-bottom: 0.3rem;
    }
    /* Beautify progress bar area */
    .progress-container {
        background-color: #f0f7ff;
        border-radius: 0.3rem;
        padding: 0.4rem;
        margin-bottom: 0.5rem;
        border: 1px solid #dce8fa;
    }
    
    /* Improve left-right alignment */
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    
    /* Ensure slider components are aligned */
    .stSlider > div {
        padding-left: 0 !important;
        padding-right: 0 !important;
    }
    
    /* Reduce chart outer margins */
    .stPlotlyChart > div, .stImage > img {
        margin: 0 auto !important;
        padding: 0 !important;
    }
    
    /* Make sidebar more compact */
    section[data-testid="stSidebar"] div.stMarkdown p {
        margin-bottom: 0.2rem;
    }
    
    /* More compact titles */
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        margin-top: 0.2rem;
        margin-bottom: 0.2rem;
    }
    
    /* Make result areas more compact */
    .results-container > div {
        margin-bottom: 0.4rem !important;
    }
</style>
""", unsafe_allow_html=True)

# Load saved random forest model
@st.cache_resource
def load_model():
    try:
        model = joblib.load('rf2.pkl')
        # Add model information
        if hasattr(model, 'n_features_in_'):
            st.session_state['model_n_features'] = model.n_features_in_
            st.session_state['model_feature_names'] = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else None
        return model
    except Exception as e:
        st.error(f"⚠️ Model file 'rf.pkl' loading error: {str(e)}. Please ensure the model file is in the correct location.")
        return None

model = load_model()

# Sidebar configuration and debug information
with st.sidebar:
    # Add font status information
    if font_downloaded:
        st.success("✅ Successfully downloaded and installed Source Han Sans")
    else:
        st.warning(f"⚠️ Unable to load Chinese font: {error_message}")
        st.info("Chinese characters in charts may not display correctly")
        
    st.markdown("### Model Information")
    if model is not None and hasattr(model, 'n_features_in_'):
        st.info(f"Model expected feature count: {model.n_features_in_}")
        if hasattr(model, 'feature_names_in_'):
            expected_features = model.feature_names_in_
            st.write("Model expected feature list:", expected_features)
    
    st.markdown("---")
    st.markdown("### Application Instructions")
    st.markdown("""
    This application is built based on the Random Forest algorithm, analyzing key clinical features of gastric cancer patients to predict the risk of death within three years after surgery.

    **How to use:**
    1. Input patient features on the right side
    2. Click the "Start Prediction" button
    3. View prediction results and explanations
    """)

# Feature range definitions
feature_ranges = {
    "Intraoperative Blood Loss": {"type": "numerical", "min": 0.000, "max": 800.000, "default": 50, 
                                 "description": "Intraoperative Blood Loss (ml)", "unit": "ml"},
    "CEA": {"type": "numerical", "min": 0, "max": 150.000, "default": 8.68, 
           "description": "CEA", "unit": "ng/ml"},
    "Albumin": {"type": "numerical", "min": 1.0, "max": 80.0, "default": 38.60, 
               "description": "Albumin", "unit": "g/L"},
    "TNM Stage": {"type": "categorical", "options": [1, 2, 3, 4], "default": 2, 
                 "description": "TNM Stage", "unit": ""},
    "Age": {"type": "numerical", "min": 25, "max": 90, "default": 76, 
           "description": "Age", "unit": "Years"},
    "Max Tumor Diameter": {"type": "numerical", "min": 0.2, "max": 20, "default": 4, 
                          "description": "Max Tumor Diameter", "unit": "cm"},
    "Lymphovascular Invasion": {"type": "categorical", "options": [0, 1], "default": 1, 
                              "description": "Lymphovascular Invasion (0=NO, 1=YES)", "unit": ""},
}

# Feature order definition - ensure consistency with model training order
if model is not None and hasattr(model, 'feature_names_in_'):
    feature_input_order = list(model.feature_names_in_)
    feature_ranges_ordered = {}
    for feature in feature_input_order:
        if feature in feature_ranges:
            feature_ranges_ordered[feature] = feature_ranges[feature]
        else:
            # Features required by model but not defined in UI
            with st.sidebar:
                st.warning(f"Model requires feature '{feature}' but it's not defined in UI")
    
    # Check features defined in UI but not required by model
    for feature in feature_ranges:
        if feature not in feature_input_order:
            with st.sidebar:
                st.warning(f"Feature '{feature}' defined in UI is not required by model")
    
    # Use sorted feature dictionary
    feature_ranges = feature_ranges_ordered
else:
    # If model doesn't have feature_names_in_ attribute, use original order
    feature_input_order = list(feature_ranges.keys())

# Application title and description
st.markdown('<h1 class="main-header">Gastric Cancer Postoperative Three-Year Survival Prediction Model</h1>', unsafe_allow_html=True)

# Create two-column layout, adjusted to more appropriate proportions
col1, col2 = st.columns([3.5, 6.5], gap="small")

with col1:
    st.markdown('<div class="section-container">', unsafe_allow_html=True)
    st.markdown('<h2 class="sub-header">Patient Feature Input</h2>', unsafe_allow_html=True)
    
    # Dynamically generate input items - more compact layout
    feature_values = {}
    
    for feature in feature_input_order:
        properties = feature_ranges[feature]
        
        # Display feature description - generate different help text based on variable type
        if properties["type"] == "numerical":
            help_text = f"{properties['description']} ({properties['min']}-{properties['max']} {properties['unit']})"
            
            # Create slider for numerical variables - use more compact layout
            value = st.slider(
                label=f"{feature}",
                min_value=float(properties["min"]),
                max_value=float(properties["max"]),
                value=float(properties["default"]),
                step=0.1,
                help=help_text,
                # Make layout more compact
            )
        elif properties["type"] == "categorical":
            # For categorical variables, only use description as help text
            help_text = f"{properties['description']}"
            
            # Create radio buttons for categorical variables
            if feature == "TNM Stage":
                options_display = {1: "Stage I", 2: "Stage II", 3: "Stage III", 4: "Stage IV"}
                value = st.radio(
                    label=f"{feature}",
                    options=properties["options"],
                    format_func=lambda x: options_display[x],
                    help=help_text,
                    horizontal=True
                )
            elif feature == "Lymphovascular Invasion":
                options_display = {0: "No", 1: "Yes"}
                value = st.radio(
                    label=f"{feature}",
                    options=properties["options"],
                    format_func=lambda x: options_display[x],
                    help=help_text,
                    horizontal=True
                )
            else:
                value = st.radio(
                    label=f"{feature}",
                    options=properties["options"],
                    help=help_text,
                    horizontal=True
                )
                
        feature_values[feature] = value
    
    # Prediction button
    predict_button = st.button("Start Prediction", help="Click to generate prediction results")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    if predict_button and model is not None:
        st.markdown('<div class="results-container">', unsafe_allow_html=True)
        st.markdown('<h2 class="sub-header">Prediction Results</h2>', unsafe_allow_html=True)
        
        # Prepare model input
        features_df = pd.DataFrame([feature_values])
        
        # Ensure feature order matches model training
        if hasattr(model, 'feature_names_in_'):
            # Check if all required features have values
            missing_features = [f for f in model.feature_names_in_ if f not in features_df.columns]
            if missing_features:
                st.error(f"Missing model required features: {missing_features}")
                st.stop()
            
            # Reorder features according to model training feature order
            features_df = features_df[model.feature_names_in_]
        
        # Convert to numpy array
        features_array = features_df.values
        
        with st.spinner("Calculating prediction results..."):
            try:
                # Model prediction
                predicted_class = model.predict(features_array)[0]
                predicted_proba = model.predict_proba(features_array)[0]
                
                # Extract predicted class probability
                death_probability = predicted_proba[1] * 100  # Assume 1 represents death class
                survival_probability = 100 - death_probability
                
                # Create probability display - further reduce size
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = death_probability,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "", 'font': {'size': 14, 'family': 'sans-serif', 'color': 'black', 'weight': 'bold'}},
                    gauge = {
                        'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue", 'tickfont': {'color': 'black', 'size': 9}},
                        'bar': {'color': "darkblue"},
                        'bgcolor': "white",
                        'borderwidth': 1,
                        'bordercolor': "gray",
                        'steps': [
                            {'range': [0, 30], 'color': 'green'},
                            {'range': [30, 70], 'color': 'orange'},
                            {'range': [70, 100], 'color': 'red'}],
                        'threshold': {
                            'line': {'color': "red", 'width': 2},
                            'thickness': 0.6,
                            'value': death_probability}}))
                
                fig.update_layout(
                    height=160,  # Further reduce height
                    margin=dict(l=5, r=5, t=5, b=5),  # Reduce top margin
                    paper_bgcolor="white",
                    plot_bgcolor="white",
                    font={'family': 'sans-serif', 'color': 'black', 'size': 11},
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Create risk category display
                risk_category = "Low Risk"
                risk_color = "green"
                if death_probability > 30 and death_probability <= 70:
                    risk_category = "Medium Risk"
                    risk_color = "orange"
                elif death_probability > 70:
                    risk_category = "High Risk"
                    risk_color = "red"
                
                # Display risk category and probability - use light background instead of white
                st.markdown(f"""
                <div style="text-align: center; margin: -0.2rem 0 0.3rem 0;">
                    <span style="font-size: 1.1rem; font-family: system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif; color: {risk_color}; font-weight: bold;">
                        {risk_category}
                    </span>
                </div>
                """, unsafe_allow_html=True)
                
                # Display specific probability values - put in light background container
                st.markdown(f"""
                <div class="progress-container">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 0.2rem;">
                        <div style="text-align: center; width: 48%;">
                            <div style="font-size: 0.9rem; font-weight: bold; color: #1E3A8A;">Three-Year Survival Probability</div>
                            <div style="font-size: 1.1rem; font-weight: bold; color: #10B981;">{survival_probability:.1f}%</div>
                        </div>
                        <div style="text-align: center; width: 48%;">
                            <div style="font-size: 0.9rem; font-weight: bold; color: #1E3A8A;">Three-Year Death Risk</div>
                            <div style="font-size: 1.1rem; font-weight: bold; color: #EF4444;">{death_probability:.1f}%</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Add SHAP visualization section - reduce spacing
                st.markdown('<hr style="margin:0.3rem 0;">', unsafe_allow_html=True)
                st.markdown('<h2 class="sub-header">Prediction Result Explanation</h2>', unsafe_allow_html=True)
                
                try:
                    with st.spinner("Generating SHAP explanation plot..."):
                        # Use latest version of SHAP API, adopt most concise and compatible approach
                        explainer = shap.Explainer(model)
                        
                        # Calculate SHAP values
                        shap_values = explainer(features_df)
                        
                        # Extract feature names and SHAP values
                        feature_names = list(features_df.columns)
                        
                        # Create a mapping dictionary, mapping original feature names to labels containing feature values
                        feature_labels_with_values = {}
                        for feature in feature_names:
                            if feature in feature_values:
                                value = feature_values[feature]
                                # Handle categorical features
                                if feature == "TNM Stage":
                                    value_display = f"Stage {int(value)}"
                                elif feature == "Lymphovascular Invasion":
                                    value_display = "Yes" if value == 1 else "No"
                                else:
                                    value_display = f"{value}"
                                feature_labels_with_values[feature] = f"{value_display} = {feature}"
                            else:
                                feature_labels_with_values[feature] = feature
                        
                        # Use labels with feature values to replace original feature names
                        features_renamed = {}
                        for i, feature in enumerate(feature_names):
                            features_renamed[i] = feature_labels_with_values[feature]
                        
                        # Set matplotlib chart style
                        plt.style.use('default')
                        
                        # Ensure font settings are correct
                        if font_downloaded and font_path:
                            # Set font again, ensure font configuration is correct before plotting
                            plt.rcParams['font.sans-serif'] = ['Source Han Sans SC', 'DejaVu Sans', 'Arial']
                        else:
                            # Use common Chinese fonts as alternatives
                            plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 
                                                            'Noto Sans CJK JP', 'Noto Sans CJK SC', 'WenQuanYi Micro Hei',
                                                            'DejaVu Sans', 'Arial']
                        
                        plt.rcParams['axes.unicode_minus'] = False
                        plt.rcParams['font.family'] = 'sans-serif'
                        
                        plt.figure(figsize=(10, 6), dpi=100, facecolor='white')
                        
                        # Select plotting method based on SHAP value type
                        if hasattr(shap_values, 'values') and len(shap_values.values.shape) > 2:
                            # Multi-class case - select second class (usually positive class/death class)
                            shap_obj = shap_values[0, :, 1]
                        else:
                            # Binary classification or regression case
                            shap_obj = shap_values[0]
                        
                        # Generate SHAP waterfall plot
                        shap_waterfall = shap.plots.waterfall(
                            shap_obj,
                            max_display=7,
                            show=False
                        )
                        
                        # Add title
                        plt.title("Feature Impact on Prediction", fontsize=14, fontweight='bold')
                        
                        # Adjust layout
                        plt.tight_layout()
                        
                        # Save and display chart
                        plt.savefig("shap_waterfall_plot.png", dpi=200, bbox_inches='tight')
                        plt.close()
                        st.image("shap_waterfall_plot.png")
                        
                        # Add brief explanation - more compact, use light background
                        st.markdown("""
                        <div style="background-color: #f0f7ff; padding: 5px; border-radius: 3px; margin-top: 3px; font-size: 0.8rem; border: 1px solid #dce8fa;">
                          <p style="margin:0"><strong>Chart Explanation:</strong> Red bars indicate features that increase death risk, blue bars indicate features that decrease death risk. Values represent the contribution size to prediction results.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                except Exception as shap_error:
                    st.error(f"Error generating SHAP plot: {str(shap_error)}")
                    st.warning("Unable to generate SHAP explanation plot, please contact technical support.")
                
            except Exception as e:
                st.error(f"Error occurred during prediction: {str(e)}")
                st.warning("Please check if input data matches model expected features, or contact developers for support.")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        # When prediction button is not clicked, don't display anything
        pass

# Add footer disclaimer
st.markdown("""
<div class="disclaimer">
    <p>📋 Disclaimer: This prediction tool is for clinical reference only and cannot replace professional medical judgment. Prediction results should be comprehensively evaluated in combination with the patient's complete clinical situation.</p>
    <p>© 2025 | Development Version v1.1.0</p>
</div>
""", unsafe_allow_html=True)  