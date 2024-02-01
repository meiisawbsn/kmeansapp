import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

st.title("K-MEANS")

def plot_elbow_method(selected_data):
    wcss = []
    for i in range(1, 10):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=None)
        kmeans.fit(selected_data)
        wcss.append(kmeans.inertia_)

    plt.plot(range(1, 10), wcss)
    plt.title('Elbow Method')
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS')  # Within cluster sum of squares
    return plt

with st.sidebar:
    selected = option_menu("MENU", ["Home",'Clustering','About'], 
        icons=['house', 'gear','envelope'], menu_icon="cast", default_index=1)
if selected == "Home":
    st.title(f"{selected}")
if selected == "Clustering":
    st.title(f"K-Mean Clustering for BANSOS")
if selected == "About":
    st.title(f"It'Me")

upFile = st.file_uploader("Choose a File", type=["csv"])
if upFile is not None:
    dataframe = pd.read_csv(upFile,delimiter=';')
    st.write(dataframe)

    # st.write(dataframe.isna().sum())

    # Create a selectbox for column selection
    selected_columns = st.multiselect('Select a column', dataframe.columns)
    if selected_columns :
        selected_data = dataframe[selected_columns]
        st.write(selected_data)
        
        fig = plot_elbow_method(selected_data)
        st.pyplot(fig)
        
        kmeans = KMeans(n_clusters = 2, init = 'k-means++', random_state = None)
        y_kmeans = kmeans.fit_predict(selected_data)
        silhouette_score_average = silhouette_score(selected_data, y_kmeans)
        st.write("silhouette Score 2 kluster =", silhouette_score_average)
