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

upFile = st.file_uploader("Choose a File")
# Sample dataframe
data = {
    'Nama': ['Alice', 'Bob', 'Charlie'],
    'Umur': [30, 25, 35],
    'Gender': ['Female', 'Male', 'Male'],
}

df = pd.DataFrame(data)

# Display dataframe
st.write(df)

# Create a selectbox for column selection
selected_column = st.multiselect('Select a column', df.columns)
st.button("Clustering")