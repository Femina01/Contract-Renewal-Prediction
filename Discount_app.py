import streamlit as st
import pandas as pd
import pickle

def decode_one_hot(row, prefix):
    for col in row.index:
        if col.startswith(prefix) and row[col] == 1:
            return col.replace(prefix, "")
    return "Unknown"


model = pickle.load(open("model.pkl", "rb"))
features = pickle.load(open("features.pkl", "rb"))

st.title("Contract Renewal Prediction")

file = st.file_uploader("Upload CSV")

if file:
    df = pd.read_csv(file)

    # Validate columns
    required_cols = ["Contract_Start", "Contract_End", "Last_Year_Discount", "Previous_Contracts"]

    for col in required_cols:
        if col not in df.columns:
            st.error(f"Missing column: {col}")
            st.stop()

    # Feature Engineering
    df["Contract_Start"] = pd.to_datetime(df["Contract_Start"])
    df["Contract_End"] = pd.to_datetime(df["Contract_End"])

    df["Contract_Duration"] = (df["Contract_End"] - df["Contract_Start"]).dt.days

    df["Is_Long_Contract"] = (df["Contract_Duration"] > 365).astype(int)
    df["High_Discount"] = (df["Last_Year_Discount"] > 15).astype(int)
    df["Loyal_Customer"] = (df["Previous_Contracts"] >= 3).astype(int)

    df = df.drop(["Contract_Start", "Contract_End"], axis=1)

    # Encoding
    df = pd.get_dummies(df)
    df = df.reindex(columns=features, fill_value=0)

    # Prediction
    prob = model.predict_proba(df)[:, 1]

    df["Renewal_Probability"] = prob
df["Suggested_Discount"] = (1 - prob) * 20

# Create user-friendly output
output_list = []

for i in range(len(df)):
    row = df.iloc[i]

    customer_id = decode_one_hot(row, "Customer_ID_")
    serial_no = decode_one_hot(row, "Serial_No_")
    product_type = decode_one_hot(row, "Product_Type_")
    contract_status = decode_one_hot(row, "Contract_Status_")
    support_package = decode_one_hot(row, "Support_Package_")
    warranty = decode_one_hot(row, "Warranty_")

    output_list.append({
        "Customer_ID": customer_id,
        "Serial_No": serial_no,
        "Product_Type": product_type,
        "Contract_Status": contract_status,
        "Support_Package": support_package,
        "Warranty": warranty,
        "Renewal_Probability": round(row["Renewal_Probability"], 2),
        "Suggested_Discount (%)": round(row["Suggested_Discount"], 2)
    })

# Convert to DataFrame
clean_df = pd.DataFrame(output_list)

# Show clean output
st.subheader("📊 Clean Prediction Output")
st.dataframe(clean_df)

st.success("Thank you for using the Contract Renewal Prediction app! 😊")
