# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeRegressor
import streamlit as st
from streamlit_option_menu import option_menu
import warnings
warnings.simplefilter("ignore")
from PIL import Image


st.set_page_config(layout="wide")
def setting_bg(background_image_url):
        st.markdown(f""" 
        <style>
            .stApp {{
                background: url('{background_image_url}') no-repeat center center fixed;
                background-size: cover;
                transition: background 0.5s ease;
            }}
            h1, h2, h3, h4, h5, h6 {{
                color: #f3f3f3;
                font-family: 'Roboto', sans-serif;
            }}
            .stButton>button {{
                color: #4e4376;
                background-color: #f3f3f3;
                transition: all 0.3s ease-in-out;
            }}
            .stButton>button:hover {{
                color: #f3f3f3;
                background-color: #2b5876;
            }}
            .stTextInput>div>div>input {{
                color: #4e4376;
                background-color: #f3f3f3;
            }}
        </style>
        """, unsafe_allow_html=True)

# Example usage with a background image URL
background_image_url = "https://miro.medium.com/v2/resize:fit:735/0*lAkevA6upQBq-NCk.jpg"
setting_bg(background_image_url)

st.markdown("""<div style='border:5px solid black; background-color:yellow; padding:10px;'> 
            <h1 style='text-align:center; color:red;'>Loan Default Prediction</h1> </div>""", unsafe_allow_html=True)


with st.sidebar:
    selected = option_menu(None, ["Home","Menu"], 
                    icons=["home","Menu"],
                    default_index=0,
                    orientation="vertical",  # Set orientation to vertical
                    styles={"nav-link": {"font-size": "25px", "text-align": "centre", "margin": "0px", "--hover-color": "#AB63FA", "transition": "color 0.3s ease, background-color 0.3s ease"},
                            "icon": {"font-size": "25px"},
                            "container" : {"max-width": "6000px", "padding": "10px", "border-radius": "5px","border": "5px solid black"},
                            "nav-link-selected": {"background-color": "red", "color": "white"}} )

if selected == "Home":
    st.markdown("## :green[**Overview :**] :red[Loan Default Prediction: is a data-driven approach to assess the risk associated with lending money to individuals. It involves analyzing historical data of loan applicants to predict the likelihood of future borrowers defaulting on their loans. By examining factors such as credit scores, income levels, debt-to-income ratios, employment status, and other relevant variables, predictive models are built using machine learning algorithms. These models provide valuable insights to financial institutions, helping them make informed decisions about approving or denying loan applications and setting appropriate interest rates. Ultimately, the goal of loan default prediction is to minimize financial losses for lenders by identifying and mitigating potential risks associated with lending.]")
            

if selected  == "Menu":
        # Define unique values for select boxes
        Gender = ['Male', 'Female']
        EmploymentStatus = ['Employed', 'Unemployed']
        Location = ['Suburban', 'Urban', 'Rural']


        # Streamlit app title
        st.title(":red[Loan_Default prediction:]")


        st.sidebar.title("Loan Details")
        Gender= st.sidebar.selectbox("Gender", options=Gender)
        EmploymentStatus = st.sidebar.selectbox("Employee Status", options=EmploymentStatus)
        Location = st.sidebar.selectbox("Location", options=Location)

        Age = st.sidebar.number_input("Age:", min_value=18, max_value=64, value=18)
        Income = st.sidebar.number_input("Income:", min_value=24156, max_value=97722, value=24156)
        Credit_Score= st.sidebar.number_input("Credit_Score:", min_value=250, max_value=850, value=250)
        Debt_to_Income_Ratio= st.sidebar.number_input("Debt_to_Income_Ratio:", min_value=0.0001, max_value=0.9999, value=0.0001)
        Existing_Loan_Balance= st.sidebar.number_input("Existing_Loan_Balance:", min_value=0, max_value=50000, value=0)
        Loan_Amount= st.sidebar.number_input("Loan_Amount:", min_value=5000, max_value=50000, value=5000)
        Interest_Rate= st.sidebar.number_input("Interest_Rate:", min_value=3, max_value=20, value=3)
        Loan_Duration_Months= st.sidebar.number_input("Loan_Duration_Months:", min_value=12, max_value=71, value=12)

        #'Age','Income','Credit_Score','Debt_to_Income_Ratio','Existing_Loan_Balance','Loan_Amount','Interest_Rate','Loan_Duration_Months']].values, x1, x2, x3),
        if st.sidebar.button("Predict Loan Default status"):
                input_data = pd.DataFrame({
                        'Age': [Age],
                        'Income': [Income],
                        'Credit_Score': [Credit_Score],
                        'Debt_to_Income_Ratio':[Debt_to_Income_Ratio],
                        'Existing_Loan_Balance': [Existing_Loan_Balance],
                        'Loan_Amount': [Loan_Amount],
                        'Interest_Rate': [Interest_Rate],
                        'Loan_Duration_Months': [Loan_Duration_Months],
                        'Gender': [Gender],
                        'EmploymentStatus': [EmploymentStatus],
                        'Location': [Location]})   
                
        
                import pickle

                with open(r'model_gb.pkl', 'rb') as file:
                        model_gb = pickle.load(file)
                with open(r'scaler_gb.pkl', 'rb') as file:
                        scaler_gb = pickle.load(file)
                with open(r'ohe.pkl', 'rb') as file:
                        ohe = pickle.load(file)
                with open(r'ohe2.pkl', 'rb') as file:
                        ohe2 = pickle.load(file)
                with open(r'ohe3.pkl', 'rb') as file:
                        ohe3 = pickle.load(file)


                from sklearn.preprocessing import LabelEncoder
                label_encoder = LabelEncoder()
                label_encoder.fit(["Non-Default", "Default"])   
                def decode_labels(encoded_data, label_encoder):
                        decoded_data = label_encoder.inverse_transform(encoded_data)
                        return decoded_data   
                
                
                new_sample = np.array([[Age,float(Income),Credit_Score,float(Debt_to_Income_Ratio),float(Existing_Loan_Balance),float(Loan_Amount),float(Interest_Rate),Loan_Duration_Months, Gender, EmploymentStatus, Location]])
                new_sample_gender = ohe.transform(new_sample[:, [8]]).toarray()
                new_sample_employeeStatus = ohe2.transform(new_sample[:, [9]]).toarray()
                new_sample_Location = ohe3.transform(new_sample[:, [10]]).toarray()
                new_sample = np.concatenate((new_sample[:, [0, 1, 2, 3, 4, 5, 6, 7]], new_sample_gender, new_sample_employeeStatus, new_sample_Location), axis=1)
                # Scaling numerical features
                new_sample1 = scaler_gb.transform(new_sample)
                new_pred = model_gb.predict(new_sample1).reshape(-1)
                # Decode the predicted labels
                decoded_pred = decode_labels(new_pred, label_encoder)
                decoded_pred_str = decoded_pred[0] 

                st.markdown(f'<div style="background-color: white; border: 2px solid black; border-radius: 5px; padding: 10px; font-size: 32px; color: red;">Loan_default_prdiction is: <span style="color: darkgreen;">{decoded_pred_str}</span></div>', unsafe_allow_html=True)


