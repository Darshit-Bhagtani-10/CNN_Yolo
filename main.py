import streamlit as st

# Import the scripts from the correct locations
import Classfication  # Import the module directly
import Pretrained     # Import the module directly

# Sidebar for navigation
st.sidebar.title("Navigation")
choice = st.sidebar.selectbox("Choose a project", ["Classfication Model", "Pretrained Model"])  # Changed order here

# Load the selected project
if choice == "Classfication Model":
    Classfication.run()  # Assumes run() is defined in Classfication.py
elif choice == "Pretrained Model":
    Pretrained.run()  # Assumes run() is defined in Pretrained.py