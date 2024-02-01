import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd

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
    if selected_columns is not None:
        selected_data = dataframe[selected_columns]
        st.write(selected_data)
        st.button("Clustering")