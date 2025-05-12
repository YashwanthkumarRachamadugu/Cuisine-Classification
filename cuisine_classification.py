import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt  # For seaborn heatmap rendering in Streamlit

# Title
st.title("Cuisine Classification App")

# Upload file
uploaded_file = st.file_uploader("Upload Dataset CSV", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Drop missing cuisines
    df = df.dropna(subset=["Cuisines"])

    # Extract primary cuisine
    df["Primary_Cuisine"] = df["Cuisines"].apply(lambda x: x.split(",")[0].strip())

    # Select features
    features = df[["City", "Average Cost for two", "Price range", "Has Table booking", "Has Online delivery", "Votes"]]
    labels = df["Primary_Cuisine"]

    # Encode binary features
    features.loc[:, "Has Table booking"] = features["Has Table booking"].map({"Yes": 1, "No": 0})
    features.loc[:, "Has Online delivery"] = features["Has Online delivery"].map({"Yes": 1, "No": 0})

    # One-hot encode City
    features = pd.get_dummies(features, columns=["City"], drop_first=True)

    # Encode target
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(features, labels_encoded, test_size=0.2, random_state=42)

    # Train classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Predict
    y_pred = clf.predict(X_test)

    # Show classification report
    st.subheader("Classification Report")
    report = classification_report(
        y_test,
        y_pred,
        labels=le.transform(le.classes_),
        target_names=le.classes_,
        output_dict=True,
        zero_division=0
    )
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df)

    # Confusion matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred, labels=le.transform(le.classes_))
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, xticklabels=le.classes_, yticklabels=le.classes_, cmap="Blues", fmt='d', ax=ax)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)
