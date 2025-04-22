import pandas as pd
import numpy as np

def calculate_kpis(df):
    """
    Calculate key performance indicators from the data
    
    Args:
        df (pandas.DataFrame): FMCG data
        
    Returns:
        dict: Dictionary of KPIs
    """
    kpis = {}
    
    # Total warehouse capacity
    if 'WH_capacity_size' in df.columns:
        size_mapping = {'Small': 1, 'Mid': 2, 'Large': 3}
        df['capacity_numeric'] = df['WH_capacity_size'].map(size_mapping)
        kpis['total_capacity'] = df['capacity_numeric'].sum()
    
    # Number of warehouses
    kpis['warehouse_count'] = len(df)
    
    # Average product weight in tons
    if 'product_wg_ton' in df.columns:
        kpis['avg_product_weight'] = df['product_wg_ton'].mean()
    
    # Number of transport issues
    if 'transport_issue_l1y' in df.columns:
        kpis['transport_issues'] = df['transport_issue_l1y'].sum()
    
    # Number of storage issues
    if 'storage_issue_reported_l3m' in df.columns:
        kpis['storage_issues'] = df['storage_issue_reported_l3m'].sum()
    
    # Warehouse breakdown rate
    if 'wh_breakdown_l3m' in df.columns:
        kpis['wh_breakdown_rate'] = df['wh_breakdown_l3m'].sum() / len(df)
    
    return kpis

def prepare_monthly_trend_data(df):
    """
    Prepare data for monthly trend visualization
    
    Args:
        df (pandas.DataFrame): FMCG data
        
    Returns:
        pandas.DataFrame: Data prepared for visualization
    """
    # This is a placeholder - in a real scenario, we would aggregate by month
    # Since our sample data doesn't have time series, we'll simulate it
    
    # Create a simulated monthly trend based on zones
    if 'zone' in df.columns:
        zones = df['zone'].unique()
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        data = []
        for zone in zones:
            zone_count = df[df['zone'] == zone].shape[0]
            for month in months:
                # Simulate some seasonal variation
                if month in ['Oct', 'Nov', 'Dec']:
                    multiplier = 1.2  # Holiday season boost
                elif month in ['Jan', 'Feb']:
                    multiplier = 0.8  # Post-holiday slump
                else:
                    multiplier = 1.0
                
                value = zone_count * multiplier * np.random.uniform(0.8, 1.2)
                data.append({
                    'Month': month,
                    'Zone': zone,
                    'Value': value
                })
        
        return pd.DataFrame(data)
    
    return pd.DataFrame()

def prepare_category_distribution(df):
    """
    Prepare data for category distribution visualization
    
    Args:
        df (pandas.DataFrame): FMCG data
        
    Returns:
        pandas.DataFrame: Data prepared for visualization
    """
    if 'Location_type' in df.columns:
        category_counts = df['Location_type'].value_counts().reset_index()
        category_counts.columns = ['Category', 'Count']
        return category_counts
    
    return pd.DataFrame()

def prepare_regional_performance(df):
    """
    Prepare data for regional performance visualization
    
    Args:
        df (pandas.DataFrame): FMCG data
        
    Returns:
        pandas.DataFrame: Data prepared for visualization
    """
    if 'zone' in df.columns and 'product_wg_ton' in df.columns:
        regional_perf = df.groupby('zone')['product_wg_ton'].agg(['sum', 'mean']).reset_index()
        regional_perf.columns = ['Region', 'Total Weight', 'Average Weight']
        return regional_perf
    
    return pd.DataFrame()

def prepare_forecast_vs_actual(df):
    """
    Prepare data for forecast vs actual visualization
    
    Args:
        df (pandas.DataFrame): FMCG data
        
    Returns:
        pandas.DataFrame: Data prepared for visualization
    """
    # This is a placeholder - in a real scenario, we would compare forecast vs actual
    # Since our sample data doesn't have this, we'll simulate it
    
    if 'zone' in df.columns:
        zones = df['zone'].unique()
        
        data = []
        for zone in zones:
            zone_df = df[df['zone'] == zone]
            actual = zone_df['product_wg_ton'].sum() if 'product_wg_ton' in df.columns else np.random.randint(1000, 5000)
            forecast = actual * np.random.uniform(0.8, 1.2)  # Simulate forecast with some error
            
            data.append({
                'Region': zone,
                'Actual': actual,
                'Forecast': forecast
            })
        
        return pd.DataFrame(data)
    
    return pd.DataFrame()
