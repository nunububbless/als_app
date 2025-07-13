import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from sklearn.preprocessing import StandardScaler # type: ignore
from sklearn.linear_model import LogisticRegression # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.metrics import accuracy_score # type: ignore

st.set_page_config(
    page_title="ALS Speech Analysis Tool",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded"

)

# styling (STREAMLIT tutorial found on youtube + Claude AI debugging)

st.markdown("""
<style>
    .main-header {
            font-size: 3rem;
            color: blue;
            text-alight: center;
            margin-bottom: 2rem;
        }
    .sub-header {
            font-size: 1.2rem;
            color: #666;
            text-align: center;
            margin-bottom: 3rem;            
        }
    .feature-card {
        background-color: white;
        padding: 1 rem;
        border-radius: 10px;
        border-left: 5px solid blue;
            margin: 1rem 0;
    }
    .prediction-result {
        font-size: 1.5rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;       
    }
    .high-risk {
        background-color: #ffebee;
        color: #c62828;
        border: 2px solid #ef5450;
    }
    .low-risk{
        background-color: #e8f5e8;
        color: #2e7d32;
        border: 2px solid #66bb6a;        
    }
</style>
""", unsafe_allow_html=True)

# WEBSITE TITLE + desc of website
st.markdown('<h1 class="main-header"> 🎤 ALS Speech Analysis Tool</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Advanced speech pattern analysis for ALS prediction using ML (machine learning)</p>',unsafe_allow_html=True)

@st.cache_data
def load_and_prepare_data():
    import kagglehub # type: ignore
    import os
    from sklearn.preprocessing import LabelEncoder # type: ignore

    path = kagglehub.dataset_download("daniilkrasnoproshin/amyotrophic-lateral-sclerosis-als")
    df = pd.read_csv(os.path.join(path, 'Minsk2020_ALS_dataset.csv'))

    le = LabelEncoder()
    df['Sex_numeric'] = le.fit_transform(df['Sex'])

    return df

@ st.cache_resource
def train_model():
    df = load_and_prepare_data()

    # Select features 
    top_features = ['PVI_a', 'Ha(4)_{sd}', 'CCi(9)', 'PVI_i', 'Hi(1)_{rel}', 'CCi(2)', 'CCi(4)', 'Ha(6)_{rel}', 'PFR_a', 'CCa(10)']
    
    X_top = df[top_features]
    y = df['Diagnosis (ALS)']

    # Split into training and testing data
    X_train_top, X_test_top, y_train, y_test = train_test_split(X_top, y, test_size=0.3, random_state=42)


    #scale data
    scaler = StandardScaler()
    X_train_top_scaled = scaler.fit_transform(X_train_top)
    X_test_top_scaled = scaler.transform(X_test_top)

# CHANGE THIS: Use scaled data for logistic regression
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train_top_scaled, y_train)
    top_predictions = lr_model.predict(X_test_top_scaled)
    
    top_accuracy = accuracy_score(y_test, top_predictions)

    return lr_model, scaler, top_accuracy, top_features

# model load as training
lr_model, scaler, top_accuracy, top_features = train_model()

st.sidebar.header("📊 Model Information")
st.sidebar.metric("Model Accuracy", f"{top_accuracy:.1%}")
st.sidebar.info(f"This model uses {len(top_features)} speech features to predict ALS risk with  {top_accuracy:.1%} accuracy.")

st.sidebar.header("📋 About the Features")
feature_descriptions = {
    'PVI_a': 'Prosodic Variability Index - measures rhythm and timing patterns in vowel pronunciation',
    'Ha(4)_{sd}': 'Harmonic Variability - analyzes variability in harmonic patterns of speech',
    'CCi(9)': 'Consonant Clarity Index - evaluates consonant clarity and articulation quality',
    'PVI_i': 'Interval Variability - measures timing variability between sounds',
    'Hi(1)_{rel}': 'Harmonic Intensity - analyzes relative harmonic intensity patterns',
    'CCi(2)': 'Consonant Clarity Index (2) - another consonant measurement',
    'CCi(4)': 'Consonant Clarity Index (4) - consonant articulation measurement',
    'Ha(6)_{rel}': 'Harmonic Analysis (6) relative - harmonic pattern analysis',
    'PFR_a': 'Fundamental Frequency Rate - vowel frequency patterns',
    'CCa(10)': 'Consonant-Consonant Analysis (10) - consonant interaction patterns'
}

for feature, description in feature_descriptions.items():
    st.sidebar.markdown(f"**{feature}**:{description}")

col1, col2 = st.columns([2,1])

with col1:
    st.header("🎯 Speech Pattern Analaysis")

    with st.form("prediction_form"):
        st.subheader("Enter Speech Feature Values")

        # create a list titled input values that stores the users 10 features and what they say
        input_values = {}

        col_a, col_b = st.columns(2)

        with col_a:
            input_values['PVI_a'] = st.number_input(
                "PVI_a (Prosodic Variability)", 
                min_value=0.0, 
                max_value=2.0, 
                value=0.3, 
                step=0.001,
                help="Measures rhythm and timing patterns in vowel pronunciation"
            )
            
            input_values['CCi(9)'] = st.number_input(
                "CCi(9) (Consonant Clarity)", 
                min_value=0.0, 
                max_value=2.0, 
                value=0.25, 
                step=0.001,
                help="Evaluates consonant clarity and articulation quality"
            )
            
            input_values['Hi(1)_{rel}'] = st.number_input(
                "Hi(1)_{rel} (Harmonic Intensity)", 
                min_value=0.0, 
                max_value=2.0, 
                value=0.4, 
                step=0.001,
                help="Analyzes relative harmonic intensity patterns"
            )
            
            input_values['CCi(4)'] = st.number_input(
                "CCi(4) (Consonant Clarity 4)", 
                min_value=0.0, 
                max_value=2.0, 
                value=0.3, 
                step=0.001,
                help="Consonant articulation measurement"
            )
            
            input_values['CCa(10)'] = st.number_input(
                "CCa(10) (Consonant Analysis)", 
                min_value=0.0, 
                max_value=2.0, 
                value=0.35, 
                step=0.001,
                help="Consonant interaction patterns"
            )
        
        with col_b:
            input_values['Ha(4)_{sd}'] = st.number_input(
                "Ha(4)_{sd} (Harmonic Variability)", 
                min_value=0.0, 
                max_value=2.0, 
                value=0.2, 
                step=0.001,
                help="Analyzes variability in harmonic patterns of speech"
            )
            
            input_values['PVI_i'] = st.number_input(
                "PVI_i (Interval Variability)", 
                min_value=0.0, 
                max_value=2.0, 
                value=0.18, 
                step=0.001,
                help="Measures timing variability between sounds"
            )
            
            input_values['CCi(2)'] = st.number_input(
                "CCi(2) (Consonant Clarity 2)", 
                min_value=0.0, 
                max_value=2.0, 
                value=0.28, 
                step=0.001,
                help="Another consonant measurement"
            )
            
            input_values['Ha(6)_{rel}'] = st.number_input(
                "Ha(6)_{rel} (Harmonic Analysis)", 
                min_value=0.0, 
                max_value=2.0, 
                value=0.32, 
                step=0.001,
                help="Harmonic pattern analysis"
            )
            
            input_values['PFR_a'] = st.number_input(
                "PFR_a (Frequency Rate)", 
                min_value=0.0, 
                max_value=2.0, 
                value=0.26, 
                step=0.001,
                help="Vowel frequency patterns"
            )
        submitted = st.form_submit_button("Analyze Speech Patterns", use_container_width=True)

        if submitted:
            # input data
            input_data = np.array([[input_values[feature] for feature in top_features]])
            input_scaled = scaler.transform(input_data)

            # predict
            prediction = lr_model.predict(input_scaled)[0]
            probability = lr_model.predict_proba(input_scaled)[0]

            #Display results
            st.subheader("Analysis Results")

            if prediction ==1:
                st.markdown(f"""
                <div class="prediction-result high-risk">
                    ⚠️ Elevated ALS Risk Detected<br>
                    Confidence: {probability[1]:.1%}
                </div>
                """, unsafe_allow_html=True)
                        
                st.warning("⚠️ **Important**: This analysis suggests speech patterns consistent with ALS progression. Please consult with a healthcare professional for proper medical evaluation.")
                        
            else:
                st.markdown(f"""
                <div class="prediction-result low-risk">
                    ✅ Low ALS Risk Indicated<br>
                    Confidence: {probability[0]:.1%}
                </div>
                """, unsafe_allow_html=True)
                st.success("**Result**: The speech pattern analysis shows characteristics less consistent with ALS progression. This is a screening tool and should not replace professional medical diagnosis.")

            # probability breakdown
            st.subheader("Probability Breakdown")

            prob_df = pd.DataFrame({
                'Outcome':['No ALS', 'ALS Risk'],
                'Probability': [probability[0], probability[1]]
            })

            fig = px.bar(prob_df, x='Outcome', y='Probability',
                        color='Outcome', color_discrete_map={'No ALS': "#667dbb", 'ALS Risk': "#F0EC0A"})
            fig.update_layout(showlegend=False, height=300)
            st.plotly_chart(fig, use_container_width=True)

with col2:
    st.header("🔬 Feature Importance")
    
    # Get feature importance from model coefficients
    feature_importance = abs(lr_model.coef_[0])
    importance_df = pd.DataFrame({
        'Feature': top_features,
        'Importance': feature_importance
    }).sort_values('Importance', ascending=True)
    
    # Create horizontal bar chart
    fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                 title="Feature Importance in ALS Prediction")
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    st.header("📈 Model Performance")
    st.metric("Overall Accuracy", f"{top_accuracy:.1%}")
    
    # Sample data button
    if st.button("📝 Load Sample Data", use_container_width=True):
        st.session_state.sample_loaded = True
        st.success("Sample data loaded! Scroll up to see the filled form.")

# footer
st.markdown("---")
