import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

st.title("K-MEANS")

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
        wcss = []

        for i in range(2,11):
            kmeans = KMeans(n_clusters= i, init = 'k-means++', random_state = None)
            kmeans.fit(selected_data)
            wcss.append(kmeans.inertia_)
            
        plt.plot(range(2,11), wcss)
        plt.title('Elbow Method')
        plt.xlabel('Cluster Number')
        plt.ylabel('WCSS')
        plt.show()
        
        st.button("Clustering")