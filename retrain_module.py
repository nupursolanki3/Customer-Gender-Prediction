print("🚀 Script has started.")  # Immediate confirmation

try:
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    import pickle

    def retrain_and_save_model():
        print("📁 Reading updated_data.csv...")

        df = pd.read_csv("C:/Users/nupur/Downloads/PWC Project/updated_data.csv")  
        print(f"✅ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")

        X = df.drop(columns=["gender"])  
        y = df["gender"]

        print("🔄 Preprocessing...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        print("🧠 Training model...")
        model = LogisticRegression()
        model.fit(X_scaled, y)

        print("💾 Saving model and scaler...")
        with open("best_model.pkl", "wb") as f:
            pickle.dump(model, f)

        with open("scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)

        print("✅ Model retrained and saved successfully.")

    if __name__ == "__main__":
        retrain_and_save_model()

except Exception as e:
    print(f"❌ ERROR: {e}")
