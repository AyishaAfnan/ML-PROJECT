import streamlit as st

import joblib 
model=joblib.load("Loan_Approval_Prediction_Dataset.pkl")
le1=joblib.load('le1.pkl')
le2=joblib.load('le2.pkl')
le3=joblib.load('le3.pkl')
le4=joblib.load('le4.pkl')
le5=joblib.load('le5.pkl')
le6=joblib.load('le6.pkl')
le7=joblib.load('le7.pkl')

sc=joblib.load("scaler.pkl")

st.title(" **🏦 Loan Approval Predictor** ")
st.write("**Enter the application details below to check for loan eligibility.**")

#numerical columns
Annual_Income=st.number_input("Annual Income($)")
Credit_Score=st.number_input("Credit Score")
Loan_Amount=st.number_input("Requested Loan Amount")
Loan_Amount_Term_Months=st.number_input('Loan_Amount_Term_Month')
Interest_Rate=st.number_input("Intrest Rate")
Existing_Monthly_Debt=st.number_input("Existing_Monthly_Debt")
DTI_Ratio=st.number_input("DTI_Ratio")

#categorical
st.write("**Categorical Informations:**")

Gender=st.selectbox("Gender",['Male', 'Female', 'Non-Binary'])
Married=st.selectbox("Marriage Status",['Married', 'Single', 'Divorced'])
Education=st.selectbox("Education Status",['Master', 'Bachelor', 'High School', 'Associate', 'PhD'])
Employment_Type=st.selectbox("Employment Type",['Part-time', 'Full-time','Self-employed', 'Contract','Unemployed'])
Property_Area=st.selectbox("Property_Area",['Semi-Urban', 'Rural', 'Urban'])
Dependents=st.selectbox("Dependents",['0', '1', '2','3+'])


#encoding
Gender=le1.transform([Gender])[0]
Married=le2.transform([Married])[0]
Dependents=le3.transform([Dependents])[0]
Education=le4.transform([Education])[0]
Employment_Type=le5.transform([Employment_Type])[0]
Property_Area=le7.transform([Property_Area])[0]


if st.button("Get Output➡️"):
    output=model.predict(sc.transform([[Annual_Income,Credit_Score,Loan_Amount,Loan_Amount_Term_Months,Interest_Rate,Existing_Monthly_Debt,DTI_Ratio,
                                   Gender,Married,Education,Employment_Type,Property_Area,Dependents]]))[0]
    if output==0: #Approved
       st.success("Congradulation!..Your Loan Application Approved.")
    else:
        st.error("Sorry!!!..Your Loan Application Rejected.")

                         