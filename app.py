import streamlit as st
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load dataset
data = pd.read_csv("./diabetes.csv")

# Custom CSS
st.markdown(
    """
    <style>
        body {
            background-color: #f3f4f6;
            font-family: 'Arial', sans-serif;
        }
        .header {
            background-color: #1e3a8a;
            color: #ffffff;
            text-align: center;
            padding: 20px;
            border-radius: 10px;
        }
        .content {
            color: black;
            background-color: #ffffff;
            padding: 25px;
            margin: 20px auto;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        .content h3 {
            color: #1e293b;
            margin-bottom: 15px;
        }
        .button {
            background-color: #10b981;
            color: white;
            padding: 12px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
        }
        .button:hover {
            background-color: #059669;
        }
        .result {
            padding: 15px;
            color: white;
            font-weight: bold;
            text-align: center;
            border-radius: 5px;
        }
        .result.success {
            background-color: #22c55e;
        }
        .result.danger {
            background-color: #ef4444;
        }
        .sidebar-title {
            font-size: 1.2em;
            font-weight: bold;
            color: #4b5563;
            margin-bottom: 15px;
        }
        .sidebar-note {
            font-size: 0.9em;
            color: #6b7280;
            margin-bottom: 20px;
        }
    </style>
    """, unsafe_allow_html=True)

# Header
st.markdown('<div class="header"><h1>Diabetes Risk Predictor</h1></div>', unsafe_allow_html=True)

# Sidebar Inputs
st.sidebar.markdown('<div class="sidebar-title">Fill Patient Details</div>', unsafe_allow_html=True)
st.sidebar.markdown('<div class="sidebar-note">Provide accurate data for better prediction results.</div>', unsafe_allow_html=True)
pregnancies = st.sidebar.number_input('Pregnancies', min_value=0, max_value=17, value=0)
glucose = st.sidebar.number_input('Glucose Level', min_value=0, max_value=200, value=120)
bp = st.sidebar.number_input('Blood Pressure', min_value=0, max_value=122, value=70)
skin_thickness = st.sidebar.number_input('Skin Thickness', min_value=0, max_value=100, value=20)
insulin = st.sidebar.number_input('Insulin Level', min_value=0, max_value=846, value=79)
bmi = st.sidebar.number_input('BMI', min_value=0.0, max_value=67.0, value=20.0)
dpf = st.sidebar.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=2.4, value=0.47)
age = st.sidebar.slider('Age', min_value=21, max_value=88, value=33)

user_data = pd.DataFrame({
    'Pregnancies': [pregnancies],
    'Glucose': [glucose],
    'BloodPressure': [bp],
    'SkinThickness': [skin_thickness],
    'Insulin': [insulin],
    'BMI': [bmi],
    'DiabetesPedigreeFunction': [dpf],
    'Age': [age]
})

# Main Content: About the App
st.markdown('<div class="content"><h3>About the Predictor</h3>'
            '<p>This application leverages machine learning to predict the likelihood of diabetes based on patient data. '
            'The model is trained on real-world health data to ensure accurate and reliable results.</p>'
            '<p>By analyzing key health metrics such as glucose level, blood pressure, BMI, and more, the system generates a personalized risk assessment.</p>'
            '</div>', unsafe_allow_html=True)

# Display User Input
st.markdown('<div class="content"><h3>Patient Data Overview</h3>'
            '<p>The details you entered are summarized below. Please review them carefully before proceeding.</p></div>',
            unsafe_allow_html=True)
st.write(user_data)

# Model Training
x = data.drop(['Outcome'], axis=1)
y = data['Outcome']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

rf = RandomForestClassifier()
rf.fit(x_train, y_train)
result = rf.predict(user_data)

# Prediction Output
prediction = "High Risk of Diabetes" if result[0] == 1 else "Low Risk of Diabetes"
st.markdown(
    f'<div class="content result {"danger" if result[0] == 1 else "success"}">{prediction}</div>',
    unsafe_allow_html=True
)

# Model Accuracy
accuracy = accuracy_score(y_test, rf.predict(x_test)) * 100
st.markdown(f'<div class="content"><h3>Model Accuracy: {accuracy:.2f}%</h3>'
            '<p>The prediction accuracy of our model is based on comprehensive training using medical datasets.</p>'
            '</div>', unsafe_allow_html=True)
