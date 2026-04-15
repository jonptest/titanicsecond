import streamlit as st
import pandas as pd
import joblib

model = joblib.load("titanic_model.pkl")

st.title("Titanic Survival Predictor")
st.write("Enter passenger details to predict the person's survival")

st.sidebar.header("Passenger Details")

pclass = st.sidebar.selectbox("Ticket Class (1st, 2nd, 3rd)", [1, 2, 3])
sex = st.sidebar.selectbox("Gender", ["male", "female"])
age = st.sidebar.slider("Age", 0, 100, 25)
sibsp = st.sidebar.number_input("Siblings/Spouses Aboard", 0, 10, 0)
parch = st.sidebar.number_input("Parents/Children Aboard", 0, 10, 0)

sex_encoded = 1 if sex == "male" else 0

input_data = pd.DataFrame([[pclass, age, sibsp, parch, sex_encoded]], columns = ['Pclass', 'Age', 'SibSp', 'Parch', 'Sex_male'])

if st.button("Predict Survival"):
    prediction = model.predict(input_data)[0]
    #remember probs returns two dimensional array
    #[0][1] gets second element which is the case 1 probability
    #eg [0.25, 0.75]
    probs = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.success(f"Result: **Survived**")
    else:
        st.error(f"Result: **Did Not Survived**")

    st.metric(label="Survival Probability", value = f"{probs:.2%}")