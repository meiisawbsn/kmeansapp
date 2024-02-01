import base64
import numpy as np
import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score,davies_bouldin_score

st.title("K-MEANS")
st.title("Yang belum : ")
st.title("- Euclidean Distance")
st.title("- Pembobotan")

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

def perform_clustering(selected_data, n_clusters, iteration):
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=None, n_init=iteration)
    kmeans.fit(selected_data)
    selected_data['Cluster'] = kmeans.labels_
    return selected_data

# def calculate_euclidean_score(selected_data, cluster_centers):
#     euclidean_scores = []
#     for i, data_point in selected_data.iterrows():
#         cluster_label = data_point['Cluster']
#         centroid = cluster_centers[cluster_label]
#         euclidean_distance = np.linalg.norm(data_point[selected_columns] - centroid)  # Euclidean distance formula
#         euclidean_scores.append(euclidean_distance)
#     return euclidean_scores

def map_cluster_labels(df):
    cluster_mapping = {1: 'Layak', 0: 'Tidak Layak'}
    df['HASIL'] = df['Cluster'].map(cluster_mapping)
    return df

def calculate_dbi(selected_data, labels):
    dbi_score = davies_bouldin_score(selected_data, labels)
    return dbi_score

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
            
        # k = st.slider('Select number of clusters', min_value=2, max_value=10)
        k=2
        # Specify number of iterations
        num_iterations = st.slider('Select number of iterations', min_value=1, max_value=10)

        for i in range(1, num_iterations + 1):
            clustered_data  = perform_clustering(selected_data, k, i)
        # for i in range(1, num_iterations + 1):
        #     cluster_centers  = perform_clustering(selected_data, k, i)

        kmeans = KMeans(n_clusters = 2, init = 'k-means++', random_state = None)
        y_kmeans = kmeans.fit_predict(selected_data)
        silhouette_score_average = silhouette_score(selected_data, y_kmeans)
        st.write(f"Silhouette Score = {silhouette_score_average}")

        dbi_score = calculate_dbi(selected_data, clustered_data['Cluster'])
        st.write(f"Davies-Bouldin Index = {dbi_score}")

        # euclidean_scores = calculate_euclidean_score(clustered_data, cluster_centers)
        # st.write("Euclidean Distance Scores:")
        # st.write(euclidean_scores)
      
        # Visualize results
        st.title("Clustering Result")
        clustered_data = map_cluster_labels(clustered_data)
        st.write(clustered_data)

        csv = clustered_data.to_csv(index=False).encode('utf-8')

        st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name='Clustering_result.csv',
            mime='text/csv')