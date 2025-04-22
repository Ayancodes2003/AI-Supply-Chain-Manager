import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

def generate_time_series_data(df, periods=12):
    """
    Generate time series data for visualization
    
    Args:
        df (pandas.DataFrame): Base data
        periods (int): Number of time periods to generate
        
    Returns:
        pandas.DataFrame: Generated time series data
    """
    # Since our sample data doesn't have time series, we'll simulate it
    if 'zone' in df.columns:
        zones = df['zone'].unique()
        
        # Generate dates for the past year
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        date_range = pd.date_range(start=start_date, end=end_date, periods=periods)
        
        data = []
        for zone in zones:
            zone_df = df[df['zone'] == zone]
            base_value = zone_df.shape[0] * 100  # Base value proportional to number of warehouses
            
            for date in date_range:
                # Add seasonal pattern
                month = date.month
                if month in [10, 11, 12]:  # Q4
                    seasonal_factor = 1.2
                elif month in [1, 2, 3]:  # Q1
                    seasonal_factor = 0.8
                elif month in [4, 5, 6]:  # Q2
                    seasonal_factor = 1.0
                else:  # Q3
                    seasonal_factor = 1.1
                
                # Add trend
                trend_factor = 1 + (date - start_date).days / 365 * 0.1
                
                # Add random noise
                noise = np.random.normal(0, 0.05)
                
                # Calculate value
                value = base_value * seasonal_factor * trend_factor * (1 + noise)
                
                data.append({
                    'Date': date,
                    'Zone': zone,
                    'Sales': value,
                    'Month': date.strftime('%b'),
                    'Quarter': f'Q{(date.month-1)//3+1}'
                })
        
        return pd.DataFrame(data)
    
    return pd.DataFrame()

def create_time_series_chart(df, zone=None, aggregation='month'):
    """
    Create a time series chart
    
    Args:
        df (pandas.DataFrame): Time series data
        zone (str, optional): Zone to filter by
        aggregation (str): Aggregation level ('day', 'week', 'month', 'quarter')
        
    Returns:
        plotly.graph_objects.Figure: Time series chart
    """
    if df.empty:
        return None
    
    # Filter by zone if specified
    if zone:
        df = df[df['Zone'] == zone]
    
    # Aggregate data
    if aggregation == 'day':
        df_agg = df.groupby(['Date', 'Zone'])['Sales'].sum().reset_index()
        x_col = 'Date'
    elif aggregation == 'week':
        df['Week'] = df['Date'].dt.isocalendar().week
        df_agg = df.groupby(['Week', 'Zone'])['Sales'].sum().reset_index()
        x_col = 'Week'
    elif aggregation == 'month':
        df_agg = df.groupby(['Month', 'Zone'])['Sales'].sum().reset_index()
        x_col = 'Month'
    else:  # quarter
        df_agg = df.groupby(['Quarter', 'Zone'])['Sales'].sum().reset_index()
        x_col = 'Quarter'
    
    # Create chart
    fig = px.line(
        df_agg,
        x=x_col,
        y='Sales',
        color='Zone',
        markers=True,
        title=f'Sales Trend by {aggregation.capitalize()}',
        template='plotly_white'
    )
    
    # Add range slider for date-based charts
    if aggregation == 'day':
        fig.update_layout(
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=3, label="3m", step="month", stepmode="backward"),
                        dict(count=6, label="6m", step="month", stepmode="backward"),
                        dict(step="all")
                    ])
                ),
                rangeslider=dict(visible=True),
                type="date"
            )
        )
    
    return fig

def perform_seasonal_decomposition(df, zone):
    """
    Perform seasonal decomposition of time series
    
    Args:
        df (pandas.DataFrame): Time series data
        zone (str): Zone to analyze
        
    Returns:
        plotly.graph_objects.Figure: Decomposition chart
    """
    if df.empty:
        return None
    
    # Filter by zone
    zone_df = df[df['Zone'] == zone].copy()
    
    if len(zone_df) < 4:
        return None
    
    # Sort by date
    zone_df = zone_df.sort_values('Date')
    
    # Create a simple decomposition (trend, seasonal, residual)
    # This is a simplified version - in a real scenario, we would use statsmodels
    
    # Calculate trend (moving average)
    window = min(3, len(zone_df) // 2)
    zone_df['Trend'] = zone_df['Sales'].rolling(window=window, center=True).mean()
    
    # Fill NaN values in trend
    zone_df['Trend'] = zone_df['Trend'].fillna(method='bfill').fillna(method='ffill')
    
    # Calculate seasonal component (simplified)
    zone_df['Month_Num'] = zone_df['Date'].dt.month
    monthly_avg = zone_df.groupby('Month_Num')['Sales'].transform('mean')
    overall_avg = zone_df['Sales'].mean()
    zone_df['Seasonal'] = monthly_avg - overall_avg
    
    # Calculate residual
    zone_df['Residual'] = zone_df['Sales'] - zone_df['Trend'] - zone_df['Seasonal']
    
    # Create figure with subplots
    fig = go.Figure()
    
    # Add traces
    fig.add_trace(go.Scatter(x=zone_df['Date'], y=zone_df['Sales'], mode='lines+markers', name='Original'))
    fig.add_trace(go.Scatter(x=zone_df['Date'], y=zone_df['Trend'], mode='lines', name='Trend'))
    fig.add_trace(go.Scatter(x=zone_df['Date'], y=zone_df['Seasonal'], mode='lines', name='Seasonal'))
    fig.add_trace(go.Scatter(x=zone_df['Date'], y=zone_df['Residual'], mode='lines', name='Residual'))
    
    # Update layout
    fig.update_layout(
        title=f'Time Series Decomposition for {zone}',
        xaxis_title='Date',
        yaxis_title='Value',
        template='plotly_white'
    )
    
    return fig
