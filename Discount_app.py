import streamlit as st
import pandas as pd
import pickle

# Load model and features
model = pickle.load(open("model.pkl", "rb"))
features = pickle.load(open("features.pkl", "rb"))

# App title
st.title("📊 Contract Renewal Prediction App")

# File upload
file = st.file_uploader("📂 Upload your CSV file", type=["csv"])

if file:
    df = pd.read_csv(file)

    # Validate required columns
    required_cols = ["Contract_Start", "Contract_End", "Last_Year_Discount", "Previous_Contracts"]

    for col in required_cols:
        if col not in df.columns:
            st.error(f"❌ Missing column: {col}")
            st.stop()

    # ==============================
    # Feature Engineering
    # ==============================
    df["Contract_Start"] = pd.to_datetime(df["Contract_Start"])
    df["Contract_End"] = pd.to_datetime(df["Contract_End"])

    df["Contract_Duration"] = (df["Contract_End"] - df["Contract_Start"]).dt.days

    df["Is_Long_Contract"] = (df["Contract_Duration"] > 365).astype(int)
    df["High_Discount"] = (df["Last_Year_Discount"] > 15).astype(int)
    df["Loyal_Customer"] = (df["Previous_Contracts"] >= 3).astype(int)

    # Drop date columns
    df = df.drop(["Contract_Start", "Contract_End"], axis=1)

    # Save original data for display
    original_df = df.copy()

    # ==============================
    # Encoding
    # ==============================
    df_encoded = pd.get_dummies(df)
    df_encoded = df_encoded.reindex(columns=features, fill_value=0)

    # ==============================
    # Prediction
    # ==============================
    prob = model.predict_proba(df_encoded)[:, 1]

    df_encoded["Renewal_Probability"] = prob
    df_encoded["Suggested_Discount"] = (1 - prob) * 20

    # ==============================
    # Create Clean Output
    # ==============================
    output_list = []

    for i in range(len(df_encoded)):
        row_encoded = df_encoded.iloc[i]
        row_original = original_df.iloc[i]

        output_list.append({
            "Customer_ID": row_original.get("Customer_ID", "N/A"),
            "Serial_No": row_original.get("Serial_No", "N/A"),
            "Product_Type": row_original.get("Product_Type", "N/A"),
            "Contract_Status": row_original.get("Contract_Status", "N/A"),
            "Support_Package": row_original.get("Support_Package", "N/A"),
            "Warranty": row_original.get("Warranty", "N/A"),
            "Last_Year_Discount": row_original.get("Last_Year_Discount", "N/A"),
            "Previous_Contracts": row_original.get("Previous_Contracts", "N/A"),
            "Contract_Duration": row_original.get("Contract_Duration", "N/A"),
            "Is_Long_Contract": row_original.get("Is_Long_Contract", "N/A"),
            "High_Discount": row_original.get("High_Discount", "N/A"),
            "Loyal_Customer": row_original.get("Loyal_Customer", "N/A"),
            "Renewal_Probability": round(row_encoded["Renewal_Probability"], 2),
            "Suggested_Discount (%)": round(row_encoded["Suggested_Discount"], 2)
        })

    clean_df = pd.DataFrame(output_list)

    # ==============================
    # Display Output
    # ==============================
    st.subheader("📊 Clean Prediction Output")
    st.dataframe(clean_df)

    # ==============================
    # Download Option
    # ==============================
    csv = clean_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Results as CSV",
        data=csv,
        file_name="renewal_predictions.csv",
        mime="text/csv"
    )

    # ==============================
    # Insights
    # ==============================
    st.subheader("📈 Insights")

    for i in range(len(clean_df)):
        if clean_df.loc[i, "Renewal_Probability"] > 0.8:
            st.success(f"✅ {clean_df.loc[i, 'Customer_ID']} → High chance of renewal")
        else:
            st.warning(f"⚠️ {clean_df.loc[i, 'Customer_ID']} → Low chance of renewal")

    st.success("🎉 Thank you for using the Contract Renewal Prediction App!")
