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
    dataframe = pd.read_csv(upFile)
    st.write(dataframe)
    # Create a selectbox for column selection
    selected_column = st.multiselect('Select a column', dataframe.columns)

st.button("Clustering")