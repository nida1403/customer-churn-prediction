import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

st.title("Customer Churn Prediction")
st.write("Machine learning app for predicting telecom customer churn.")

train_path = Path("data/churn_data_80.csv")
test_path = Path("data/churn_data_20.csv")

if not train_path.exists() or not test_path.exists():
    st.error("Training or test file not found in the data folder.")
    st.stop()

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

train_df.columns = train_df.columns.str.strip().str.replace(" ", "_").str.lower()
test_df.columns = test_df.columns.str.strip().str.replace(" ", "_").str.lower()

train_df = train_df.drop_duplicates()
test_df = test_df.drop_duplicates()

train_df["churn"] = train_df["churn"].astype(str).str.strip().str.lower().map({
    "true": 1, "false": 0, "yes": 1, "no": 0, "1": 1, "0": 0
})
test_df["churn"] = test_df["churn"].astype(str).str.strip().str.lower().map({
    "true": 1, "false": 0, "yes": 1, "no": 0, "1": 1, "0": 0
})

train_df = train_df.dropna(subset=["churn"])
test_df = test_df.dropna(subset=["churn"])

train_df["churn"] = train_df["churn"].astype(int)
test_df["churn"] = test_df["churn"].astype(int)

X_train = train_df.drop("churn", axis=1).copy()
y_train = train_df["churn"].copy()

X_test = test_df.drop("churn", axis=1).copy()
y_test = test_df["churn"].copy()

for col in X_train.columns:
    if X_train[col].dtype == "object":
        fill_val = X_train[col].mode()[0]
        X_train[col] = X_train[col].fillna(fill_val)
        X_test[col] = X_test[col].fillna(fill_val)

        le = LabelEncoder()
        le.fit(X_train[col].astype(str))

        X_train[col] = le.transform(X_train[col].astype(str))

        test_values = X_test[col].astype(str)
        unseen_mask = ~test_values.isin(le.classes_)
        if unseen_mask.any():
            test_values = test_values.copy()
            test_values[unseen_mask] = le.classes_[0]

        X_test[col] = le.transform(test_values)
    else:
        med = X_train[col].median()
        X_train[col] = X_train[col].fillna(med)
        X_test[col] = X_test[col].fillna(med)

model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)
cm = confusion_matrix(y_test, y_pred)

col1, col2 = st.columns(2)
col1.metric("Accuracy", f"{acc:.3f}")
col2.metric("ROC-AUC", f"{auc:.3f}")

st.subheader("Confusion Matrix")
st.write(cm)

st.subheader("Feature Importance")
importance_df = pd.DataFrame({
    "feature": X_train.columns,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False).head(10)

fig, ax = plt.subplots()
ax.barh(importance_df["feature"][::-1], importance_df["importance"][::-1])
ax.set_xlabel("Importance")
ax.set_title("Top 10 Features")
st.pyplot(fig)

st.subheader("Business Value")
st.write(
    "This model helps identify customers at risk of churn so organisations can take "
    "targeted retention actions and reduce revenue loss."
)
