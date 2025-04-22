import streamlit as st
import pandas as pd
import os
from pathlib import Path
import datetime

def upload_data():
    """
    Handle data upload functionality
    
    Returns:
        pandas.DataFrame or None: Uploaded data if successful, None otherwise
    """
    st.subheader("Upload New Data")
    
    uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx", "xls"])
    
    if uploaded_file is not None:
        try:
            # Determine file type
            file_extension = uploaded_file.name.split(".")[-1]
            
            # Read the file
            if file_extension == "csv":
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            # Display preview
            st.write("Data Preview:")
            st.dataframe(df.head())
            
            # Ask for confirmation
            if st.button("Confirm Upload"):
                # Save the file
                save_uploaded_file(uploaded_file)
                
                st.success(f"File {uploaded_file.name} uploaded successfully!")
                return df
            
        except Exception as e:
            st.error(f"Error: {e}")
    
    return None

def save_uploaded_file(uploaded_file):
    """
    Save the uploaded file to the data directory
    
    Args:
        uploaded_file: Streamlit UploadedFile object
    """
    # Create uploads directory if it doesn't exist
    uploads_dir = Path("./data/uploads")
    uploads_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate a unique filename with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_extension = uploaded_file.name.split(".")[-1]
    new_filename = f"upload_{timestamp}.{file_extension}"
    
    # Save the file
    file_path = uploads_dir / new_filename
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    return file_path

def list_available_datasets():
    """
    List all available datasets in the data directory
    
    Returns:
        list: List of dataset filenames
    """
    datasets = []
    
    # Check main data directory
    data_dir = Path("./data")
    if data_dir.exists():
        for file in data_dir.glob("*.csv"):
            datasets.append(str(file.relative_to(data_dir)))
        
        for file in data_dir.glob("*.xlsx"):
            datasets.append(str(file.relative_to(data_dir)))
    
    # Check uploads directory
    uploads_dir = Path("./data/uploads")
    if uploads_dir.exists():
        for file in uploads_dir.glob("*.csv"):
            datasets.append(f"uploads/{file.name}")
        
        for file in uploads_dir.glob("*.xlsx"):
            datasets.append(f"uploads/{file.name}")
    
    return datasets

def load_selected_dataset(dataset_name):
    """
    Load a selected dataset
    
    Args:
        dataset_name (str): Name of the dataset to load
        
    Returns:
        pandas.DataFrame: Loaded dataset
    """
    file_path = Path("./data") / dataset_name
    
    # Determine file type
    file_extension = file_path.suffix.lower()
    
    # Read the file
    if file_extension == ".csv":
        df = pd.read_csv(file_path)
    else:
        df = pd.read_excel(file_path)
    
    return df
