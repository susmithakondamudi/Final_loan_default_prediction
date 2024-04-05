# Final_loan_default_prediction

"loan_default_prediction" utilizes machine learning to assess loan default likelihood by analyzing factors like credit history, income. It categorizes borrowers into risk groups, aiding lenders in decision-making, mitigating financial risks, and enabling proactive measures to maintain a healthy loan portfolio,ultimately stabilizing the financial system.

# 1. Dataset Loading and Preprocessing:

Loading the dataset containing loan applicant information.
Preprocessing steps including handling missing values, converting categorical variables to numeric format, and outlier detection using Isolation Forest.

# 2. Exploratory Data Analysis (EDA):

Visualizing data distribution and identifying patterns or anomalies.
Checking skewness of numerical features and identifying outliers using boxplots.

# 3. Model Building and Evaluation:

Splitting the data into training and testing sets.
Handling class imbalance using SMOTE.
Encoding categorical variables for model training.
Training three classifiers: Logistic Regression, Decision Tree, and Random Forest.
Evaluating models on both training and testing data to assess accuracy.

# 4. Hyperparameter Tuning:

Fine-tuning the hyperparameters of the Random Forest classifier using GridSearchCV to improve performance.

# 5. Model Evaluation and Performance Metrics:

Assessing the best model's performance on the test set.
Generating a confusion matrix, ROC curve, and classification report to evaluate prediction accuracy in determining loan default status.

# 6. Results:

Comparing the performance of different classifiers.
Reporting the final model's accuracy in predicting loan default status based on the provided features.

# Streamlit application for predicting loan default status based on user-inputted information. Here's a brief summary:

•	The application utilizes various Python libraries such as Pandas, NumPy, Scikit-learn, Streamlit, and PIL (Python Imaging Library).

•	It features a background image to enhance visual appeal.

•	Users can input information about loan applicants, including gender, employment status, location, age, income, credit score, debt-to-income ratio, existing loan 
  balance, loan amount, interest rate, and loan duration in months.
  
•	Upon providing the required information and clicking the "Predict Loan Default status" button, the application predicts whether the loan applicant is likely to 
  default or not.
  
•	The prediction is made using a machine learning model trained on historical data. The model is loaded from a pickle file.

•	The predicted loan default status is displayed on the interface, indicating whether the applicant is classified as "Default" or "Non-Default".

•	The interface provides navigation between the "Home" and "Menu" sections using the sidebar.
