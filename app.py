import streamlit as st
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load the diabetes dataset
data = pd.read_csv("./diabetes.csv")

# Streamlit Markdown for Styling
st.markdown(
    """
    <style>
        /* Global styles */
        body {
            background: linear-gradient(135deg, #00C9FF, #92FE9D);
            font-family: 'Poppins', sans-serif;
            margin: 0;
            padding: 0;
            color: #333;
        }

        /* Main title section */
        .main-title {
            text-align: center;
            font-size: 3em;
            font-weight: 700;
            color: white;
            padding: 50px 20px;
            background: linear-gradient(45deg, #FF416C, #FF4B2B);
            background-size: 200% 200%;
            animation: gradient 5s ease infinite;
            border-radius: 12px;
            box-shadow: 0px 4px 30px rgba(0, 0, 0, 0.2);
            margin: 40px auto;
            width: 90%;
        }

        @keyframes gradient {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

        /* Section titles */
        .section-title {
            font-size: 2.5em;
            font-weight: 600;
            color: #2C3E50;
            margin: 50px 0;
            text-align: center;
        }

        /* Subtitle styling */
        .subtitle {
            text-align: center;
            font-size: 1.5em;
            color: #7F8C8D;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0px 6px 20px rgba(0, 0, 0, 0.1);
            margin-bottom: 50px;
            width: 75%;
            margin-left: auto;
            margin-right: auto;
        }

        /* Sidebar styles */
        .sidebar {
            background-color: #fff;
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.1);
            margin: 30px 0;
        }

        /* Custom input styling */
        .stNumberInput, .stTextInput, .stSlider {
            border-radius: 10px;
            border: 2px solid #00C9FF;
            padding: 10px;
            width: 100%;
            box-sizing: border-box;
            margin-bottom: 20px;
            font-size: 1.1em;
        }

        /* Button styling */
        .stButton {
            font-size: 1.1em;
            padding: 15px 40px;
            background-color: #FF416C;
            color: white;
            border-radius: 30px;
            border: none;
            box-shadow: 0px 6px 20px rgba(0, 0, 0, 0.15);
            transition: background-color 0.3s ease;
        }

        .stButton:hover {
            background-color: #FF4B2B;
            cursor: pointer;
        }

        /* Prediction result */
        .prediction-result {
            text-align: center;
            font-size: 2.2em;
            font-weight: bold;
            padding: 30px;
            background: #fff;
            border-radius: 12px;
            margin-top: 40px;
            width: 75%;
            margin-left: auto;
            margin-right: auto;
            box-shadow: 0px 6px 30px rgba(0, 0, 0, 0.2);
        }

        .prediction-result.danger {
            color: #FF4136;
        }

        .prediction-result.success {
            color: #28a745;
        }

        /* Progress Bar */
        .progress-bar {
            margin: 20px 0;
        }

        /* Cards for user data */
        .user-data-card {
            background: #fff;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0px 6px 15px rgba(0, 0, 0, 0.1);
        }

        /* Separator */
        .separator {
            border: 0;
            height: 2px;
            background: #00C9FF;
            margin: 40px auto;
            width: 80%;
            border-radius: 2px;
        }

    </style>
    """, unsafe_allow_html=True)

# Main Title
st.markdown('<div class="main-title">Diabetes Prediction App</div>', unsafe_allow_html=True)

# Subtitle
st.markdown('<div class="subtitle">Predict the likelihood of diabetes based on health data inputs.</div>', unsafe_allow_html=True)

# Sidebar Input Form for User Data
st.sidebar.header('Enter Patient Data')
st.sidebar.write("Fill in the details below for diabetes prediction:")

def calc():
    pregnancies = st.sidebar.number_input('Pregnancies', min_value=0, max_value=17, value=3)
    bp = st.sidebar.number_input('Blood Pressure', min_value=0, max_value=122, value=70)
    bmi = st.sidebar.number_input('BMI', min_value=0, max_value=67, value=20)
    glucose = st.sidebar.number_input('Glucose', min_value=0, max_value=200, value=120)
    skinthickness = st.sidebar.number_input('Skin Thickness', min_value=0, max_value=100, value=20)
    dpf = st.sidebar.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=2.4, value=0.47)
    insulin = st.sidebar.number_input('Insulin', min_value=0, max_value=846, value=79)
    age = st.sidebar.number_input('Age', min_value=21, max_value=88, value=33)

    output = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': bp,
        'SkinThickness': skinthickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': dpf,
        'Age': age
    }
    report_data = pd.DataFrame(output, index=[0])
    return report_data

user_data = calc()

# Display User Data Summary
st.markdown('<div class="section-title">Patient Data Summary</div>', unsafe_allow_html=True)
st.write(user_data)

# Train Model
x = data.drop(['Outcome'], axis=1)
y = data['Outcome']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

st.markdown('<hr class="separator">', unsafe_allow_html=True)

# Display Progress Bar while Training Model
progress = st.progress(0)
rf = RandomForestClassifier()
rf.fit(x_train, y_train)
progress.progress(100)

# Prediction Result
result = rf.predict(user_data)
prediction = 'You are not Diabetic' if result[0] == 0 else 'You are Diabetic'

# Display Prediction Result
st.markdown(f"<div class='prediction-result {'danger' if result[0] == 1 else 'success'}'>"
            f"{prediction}</div>", unsafe_allow_html=True)

# Display Model Accuracy
accuracy = accuracy_score(y_test, rf.predict(x_test)) * 100
st.markdown(f'<div class="section-title">Model Accuracy : {accuracy:.3f}%</div>', unsafe_allow_html=True)
