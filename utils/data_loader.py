import pandas as pd
import os

def load_data():
    """
    Load FMCG data from CSV file
    
    Returns:
        pandas.DataFrame: Loaded and preprocessed data
    """
    # Path to the data file
    data_path = os.path.join('data', 'FMCG_data.csv')
    
    # Load the data
    df = pd.read_csv(data_path)
    
    # Basic preprocessing
    # Convert date columns if they exist
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
    
    return df

def get_regions(df):
    """
    Get unique regions from the data
    
    Args:
        df (pandas.DataFrame): FMCG data
        
    Returns:
        list: List of unique regions
    """
    if 'zone' in df.columns:
        return sorted(df['zone'].unique().tolist())
    return []

def get_categories(df):
    """
    Get unique product categories from the data
    
    Args:
        df (pandas.DataFrame): FMCG data
        
    Returns:
        list: List of unique product categories
    """
    # Assuming we have a category column
    if 'Location_type' in df.columns:
        return sorted(df['Location_type'].unique().tolist())
    return []

def filter_data(df, regions=None, categories=None, date_range=None):
    """
    Filter data based on selected regions, categories, and date range
    
    Args:
        df (pandas.DataFrame): FMCG data
        regions (list, optional): List of selected regions
        categories (list, optional): List of selected categories
        date_range (tuple, optional): Start and end dates
        
    Returns:
        pandas.DataFrame: Filtered data
    """
    filtered_df = df.copy()
    
    # Filter by region
    if regions and 'zone' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['zone'].isin(regions)]
    
    # Filter by category
    if categories and 'Location_type' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Location_type'].isin(categories)]
    
    # Filter by date range
    if date_range and 'Date' in filtered_df.columns:
        start_date, end_date = date_range
        filtered_df = filtered_df[(filtered_df['Date'] >= start_date) & 
                                 (filtered_df['Date'] <= end_date)]
    
    return filtered_df
