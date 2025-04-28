import pandas as pd
import streamlit as st
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.title("🌾 Crop Production Prediction Dashboard")

st.sidebar.header("📊 Enter Crop Data")
Domain = st.sidebar.number_input("Domain", min_value=0)
Area = st.sidebar.number_input("Area", min_value=0)
Item = st.sidebar.number_input("Item", min_value=0)
Unit = st.sidebar.number_input("Unit", min_value=0)
Year = st.sidebar.number_input("Year", min_value=2019, max_value=2025, step=1)
Area_harvested = st.sidebar.number_input("Area harvested (in hectares)", min_value=0.0)
Yield = st.sidebar.number_input("Yield (in tons/ha)", min_value=0.0)

try:
    model = joblib.load(r"D:\DS-Class\project\Mini_pro3\model.pk1")
    st.success("✅ Model Loaded Successfully!")
except Exception as e:
    st.error(f"❌ Error Loading Model: {e}")

if st.sidebar.button("🔮 Predict Production"):
    input_data = pd.DataFrame([[Domain, Area, Item, Unit, Year, Area_harvested, Yield]],
                              columns=['Domain', 'Area', 'Item', 'Unit', 'Year', 'Area harvested', 'Yield'])
    try:
        prediction = model.predict(input_data)[0]
        st.metric(label="Predicted Crop Production", value=f"{prediction:.2f} units")
    except Exception as e:
        st.error(f"Prediction Failed: {e}")

if st.checkbox("🛠 Train New Model"):
    st.subheader("📂 Training Data Loaded")
    data = pd.read_csv("Prediction.csv")
    st.write("Dataset Preview:", data.head())

    label_encoder = LabelEncoder()
    for col in ['Domain', 'Area', 'Item', 'Unit']:
        data[col] = label_encoder.fit_transform(data[col])
    
    X = data.drop(columns=['Production'])
    y = data['Production']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    y_pred = rf_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    
    st.write(f"**Model Performance:**")
    st.write(f"📉 Mean Absolute Error: {mae:.2f}")
    st.write(f"📈 Mean Squared Error: {mse:.2f}")
    st.write(f"📊 Root Mean Squared Error: {rmse:.2f}")
    st.write(f"📌 R2 Score: {r2:.2f}")
    
    joblib.dump(rf_model, "/mnt/data/new_model.pkl")
    st.success("✅ Model Trained and Saved as 'new_model.pkl'")

    fig, ax = plt.subplots()
    sns.scatterplot(x=y_test, y=y_pred, ax=ax)
    ax.set_xlabel("Actual Production")
    ax.set_ylabel("Predicted Production")
    ax.set_title("Actual vs Predicted Crop Production")
    st.pyplot(fig)

st.header("📖 Reference Table")
try:
    df = pd.read_csv("/mnt/data/agriculture.csv")
    st.dataframe(df.drop(columns=['Production'], errors='ignore'))
except:
    st.warning("⚠️ Reference data not found.")
