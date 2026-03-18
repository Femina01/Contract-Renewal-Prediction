import streamlit as st
import pandas as pd
import pickle

model = pickle.load(open("model.pkl", "rb"))
features = pickle.load(open("features.pkl", "rb"))

st.title("Contract Renewal Prediction")
st.write("This app predicts the likelihood that a contract will be renewed and suggests a discount percentage to improve renewal probability.")

file = st.file_uploader("Upload CSV")

if file:
    df = pd.read_csv(file)

    df["Contract_Start"] = pd.to_datetime(df["Contract_Start"])
    df["Contract_End"] = pd.to_datetime(df["Contract_End"])

    df["Contract_Duration"] = (df["Contract_End"] - df["Contract_Start"]).dt.days

    df["Is_Long_Contract"] = (df["Contract_Duration"] > 365).astype(int)
    df["High_Discount"] = (df["Last_Year_Discount"] > 15).astype(int)
    df["Loyal_Customer"] = (df["Previous_Contracts"] >= 3).astype(int)

    df = df.drop(["Contract_Start", "Contract_End"], axis=1)

    df = pd.get_dummies(df)
    df = df.reindex(columns=features, fill_value=0)

    prob = model.predict_proba(df)[:, 1]

    df["Renewal_Probability"] = prob
    df["Suggested_Discount"] = (1 - prob) * 20

    st.dataframe(df)
    st.success("Thank you for using the Contract Renewal Prediction app! 😊")