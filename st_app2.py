
import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
categorical_features = ['job', 'marital', 'education', 'default', 'housing', 'loan',
                        'contact', 'month', 'day_of_week', 'poutcome']
# Load artifacts
model = joblib.load('bank_model.pkl')
preprocessor = joblib.load('preprocessor.pkl')

# App interface
st.title('bank loan subscription Predictor')

# visualization
st.subheader("Age Distribution")
fig, ax = plt.subplots()
sns.histplot(df_cleaned['age'], kde=True, ax=ax)
st.pyplot(fig)

st.subheader("education Distribution")
fig, ax = plt.subplots()
df_cleaned['education'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax)
st.pyplot(fig)

st.subheader("laon histogram")
feature = st.selectbox("laon", df_cleaned.columns)
fig, ax = plt.subplots()
sns.histplot(df_cleaned[feature], kde=True, ax=ax)
st.pyplot(fig)

st.subheader("Correlation Heatmap")
fig, ax = plt.subplots()
sns.heatmap(df_cleaned.corr(), annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig)

# Input widgets
col1, col2 = st.columns(2)
with col1:
  age = st.slider("Age", 18, 95, 30)
  duration = st.slider("Call Duration (seconds)", 0, 5000, 100)
  campaign = st.slider("Number of Contacts During Campaign", 1, 50, 1)
  pdays = st.slider("Days Since Last Contact", -1, 999, 100)
  previous = st.slider("Number of Previous Contacts", 0, 50, 0)
  emp_var_rate = st.number_input("Employment Variation Rate", value=1.1)
  cons_price_idx = st.number_input("Consumer Price Index", value=93.2)
  cons_conf_idx = st.number_input("Consumer Confidence Index", value=-40.0)
  euribor3m = st.number_input("Euribor 3 Month Rate", value=4.86)
  nr_employed = st.number_input("Number of Employees", value=5191.0)

with col2:
  job = st.selectbox("Job", ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management',
                           'retired', 'self-employed', 'services', 'student', 'technician', 'unemployed', 'unknown'])
  marital = st.selectbox("Marital Status", ['divorced', 'married', 'single', 'unknown'])
  education = st.selectbox("Education", ['basic.4y', 'basic.6y', 'basic.9y', 'high.school',
                                       'illiterate', 'professional.course', 'university.degree', 'unknown'])
  default = st.selectbox("Has Credit in Default?", ['no', 'yes', 'unknown'])
  housing = st.selectbox("Has Housing Loan?", ['no', 'yes', 'unknown'])
  loan = st.selectbox("Has Personal Loan?", ['no', 'yes', 'unknown'])
  contact = st.selectbox("Contact Communication Type", ['cellular', 'telephone'])
  month = st.selectbox("Last Contact Month", ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])
  day_of_week = st.selectbox("Last Contact Day", ['mon', 'tue', 'wed', 'thu', 'fri'])
  poutcome = st.selectbox("Previous Outcome", ['failure', 'nonexistent', 'success'])

# Prediction logic
input_data = pd.DataFrame([{
    'age': age,
    'duration': duration,
    'campaign': campaign,
    'pdays': pdays,
    'previous': previous,
    'emp.var.rate': emp_var_rate,
    'cons.price.idx': round(cons_price_idx, 1),
    'cons.conf.idx': cons_conf_idx,
    'euribor3m': euribor3m,
    'nr.employed': nr_employed,
    'job': job,
    'marital': marital,
    'education': education,
    'default': default,
    'housing': housing,
    'loan': loan,
    'contact': contact,
    'month': month,
    'day_of_week': day_of_week,
    'poutcome': poutcome
}])

if st.button("Predict"):
    input_data[categorical_features] = input_data[categorical_features].astype(str)
    input_processed = preprocessor.transform(input_data)
    prediction = model.predict(input_processed)
    pred_prob = model.predict_proba(input_processed)[0][1]

    st.subheader("Prediction:")
    st.write("The client is likely to **subscribe**." if prediction[0] == 1 else " The client is **not likely to subscribe**.")

