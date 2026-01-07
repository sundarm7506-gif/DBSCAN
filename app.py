import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

st.title("DBSCAN Clustering App")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    X = df.select_dtypes(include=["int64", "float64"])
    st.write("Using numerical features only")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    eps = st.slider("eps value", 0.5, 5.0, 2.0)
    min_samples = st.slider("min_samples", 1, 10, 2)

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(X_scaled)

    df["DBSCAN_Cluster"] = clusters

    st.subheader("Clustered Data")
    st.dataframe(df.head())

    st.subheader("Cluster Distribution")
    st.write(df["DBSCAN_Cluster"].value_counts())

    if X.shape[1] >= 2:
        fig, ax = plt.subplots()
        ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters)
        ax.set_xlabel(X.columns[0])
        ax.set_ylabel(X.columns[1])
        ax.set_title("DBSCAN Clustering Visualization")
        st.pyplot(fig)
