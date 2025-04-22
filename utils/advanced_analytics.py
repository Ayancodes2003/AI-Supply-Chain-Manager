import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
import statsmodels.api as sm
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx

def forecast_demand(df, periods=3):
    """
    Forecast future demand using time series analysis
    
    Args:
        df (pandas.DataFrame): Historical data
        periods (int): Number of periods to forecast
        
    Returns:
        pandas.DataFrame: Forecast results
    """
    # This is a simplified forecasting model
    # In a real scenario, we would use more sophisticated models like ARIMA or Prophet
    
    # Simulate forecasting with a simple moving average
    if 'product_wg_ton' in df.columns and 'zone' in df.columns:
        # Group by zone
        zones = df['zone'].unique()
        
        forecast_data = []
        for zone in zones:
            zone_data = df[df['zone'] == zone]
            avg_weight = zone_data['product_wg_ton'].mean()
            
            # Generate forecast with some random variation
            for i in range(1, periods + 1):
                forecast_data.append({
                    'Zone': zone,
                    'Period': f'Period {i}',
                    'Forecasted_Weight': avg_weight * (1 + 0.05 * i * np.random.randn())
                })
        
        return pd.DataFrame(forecast_data)
    
    return pd.DataFrame()

def detect_anomalies(df, contamination=0.05):
    """
    Detect anomalies in the data using Isolation Forest
    
    Args:
        df (pandas.DataFrame): Data to analyze
        contamination (float): Expected proportion of anomalies
        
    Returns:
        pandas.DataFrame: Data with anomaly flags
    """
    if 'product_wg_ton' in df.columns:
        # Create a copy of the dataframe
        result_df = df.copy()
        
        # Select numerical features for anomaly detection
        numerical_cols = ['product_wg_ton']
        if 'transport_issue_l1y' in df.columns:
            numerical_cols.append('transport_issue_l1y')
        if 'storage_issue_reported_l3m' in df.columns:
            numerical_cols.append('storage_issue_reported_l3m')
        if 'wh_breakdown_l3m' in df.columns:
            numerical_cols.append('wh_breakdown_l3m')
        
        # Apply Isolation Forest
        model = IsolationForest(contamination=contamination, random_state=42)
        result_df['anomaly'] = model.fit_predict(result_df[numerical_cols])
        
        # Convert predictions to binary flag (1 for normal, -1 for anomaly)
        result_df['is_anomaly'] = result_df['anomaly'].apply(lambda x: 'Yes' if x == -1 else 'No')
        
        return result_df
    
    return df

def segment_warehouses(df, n_clusters=3):
    """
    Segment warehouses using K-means clustering
    
    Args:
        df (pandas.DataFrame): Warehouse data
        n_clusters (int): Number of clusters
        
    Returns:
        pandas.DataFrame: Data with cluster assignments
    """
    if 'product_wg_ton' in df.columns:
        # Create a copy of the dataframe
        result_df = df.copy()
        
        # Select features for clustering
        features = ['product_wg_ton']
        if 'transport_issue_l1y' in df.columns:
            features.append('transport_issue_l1y')
        if 'storage_issue_reported_l3m' in df.columns:
            features.append('storage_issue_reported_l3m')
        
        # Apply K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        result_df['cluster'] = kmeans.fit_predict(result_df[features])
        
        # Map cluster numbers to meaningful labels
        cluster_map = {
            0: 'High Performance',
            1: 'Medium Performance',
            2: 'Low Performance'
        }
        result_df['performance_segment'] = result_df['cluster'].map(cluster_map)
        
        return result_df
    
    return df

def create_supply_chain_network(df):
    """
    Create a supply chain network visualization
    
    Args:
        df (pandas.DataFrame): Supply chain data
        
    Returns:
        plotly.graph_objects.Figure: Network visualization
    """
    if 'zone' in df.columns and 'WH_regional_zone' in df.columns:
        # Create a graph
        G = nx.Graph()
        
        # Add nodes for zones
        zones = df['zone'].unique()
        for zone in zones:
            G.add_node(zone, type='zone')
        
        # Add nodes for regional zones
        regional_zones = df['WH_regional_zone'].unique()
        for rz in regional_zones:
            G.add_node(rz, type='regional_zone')
        
        # Add edges between zones and regional zones
        for _, row in df.iterrows():
            G.add_edge(row['zone'], row['WH_regional_zone'])
        
        # Create positions for nodes
        pos = nx.spring_layout(G, seed=42)
        
        # Create edge trace
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines')
        
        # Create node trace
        node_x = []
        node_y = []
        node_text = []
        node_color = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node)
            
            # Color nodes by type
            if G.nodes[node]['type'] == 'zone':
                node_color.append('#1f77b4')  # Blue for zones
            else:
                node_color.append('#ff7f0e')  # Orange for regional zones
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            text=node_text,
            marker=dict(
                showscale=False,
                color=node_color,
                size=10,
                line_width=2))
        
        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace],
                     layout=go.Layout(
                        title='Supply Chain Network',
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20,l=5,r=5,t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                        )
        
        return fig
    
    return None

def create_heatmap(df):
    """
    Create a heatmap of warehouse performance by region
    
    Args:
        df (pandas.DataFrame): Warehouse data
        
    Returns:
        plotly.graph_objects.Figure: Heatmap visualization
    """
    if 'zone' in df.columns and 'WH_regional_zone' in df.columns:
        # Create a pivot table
        if 'product_wg_ton' in df.columns:
            pivot_data = df.pivot_table(
                values='product_wg_ton',
                index='zone',
                columns='WH_regional_zone',
                aggfunc='mean'
            ).fillna(0)
            
            # Create heatmap
            fig = px.imshow(
                pivot_data,
                labels=dict(x="Regional Zone", y="Zone", color="Avg Product Weight (tons)"),
                x=pivot_data.columns,
                y=pivot_data.index,
                color_continuous_scale='Viridis',
                title='Regional Performance Heatmap'
            )
            
            return fig
    
    return None

def generate_optimization_recommendations(df):
    """
    Generate optimization recommendations based on data analysis
    
    Args:
        df (pandas.DataFrame): Warehouse data
        
    Returns:
        list: List of recommendations
    """
    recommendations = []
    
    # Check for warehouses with high product weight but low capacity
    if 'product_wg_ton' in df.columns and 'WH_capacity_size' in df.columns:
        high_weight_small_capacity = df[(df['product_wg_ton'] > df['product_wg_ton'].quantile(0.75)) & 
                                       (df['WH_capacity_size'] == 'Small')]
        
        if len(high_weight_small_capacity) > 0:
            recommendations.append(f"Consider upgrading {len(high_weight_small_capacity)} small warehouses with high product weight to larger capacity.")
    
    # Check for transport issues
    if 'transport_issue_l1y' in df.columns and 'zone' in df.columns:
        transport_issues_by_zone = df.groupby('zone')['transport_issue_l1y'].sum()
        problem_zones = transport_issues_by_zone[transport_issues_by_zone > transport_issues_by_zone.median()]
        
        for zone, issues in problem_zones.items():
            recommendations.append(f"Zone {zone} has {issues} transport issues. Consider reviewing transportation logistics.")
    
    # Check for storage issues
    if 'storage_issue_reported_l3m' in df.columns and 'zone' in df.columns:
        storage_issues_by_zone = df.groupby('zone')['storage_issue_reported_l3m'].sum()
        problem_zones = storage_issues_by_zone[storage_issues_by_zone > storage_issues_by_zone.median()]
        
        for zone, issues in problem_zones.items():
            recommendations.append(f"Zone {zone} has {issues} storage issues. Consider improving storage facilities.")
    
    # Check for warehouse breakdowns
    if 'wh_breakdown_l3m' in df.columns and 'zone' in df.columns:
        breakdown_by_zone = df.groupby('zone')['wh_breakdown_l3m'].sum()
        problem_zones = breakdown_by_zone[breakdown_by_zone > breakdown_by_zone.median()]
        
        for zone, breakdowns in problem_zones.items():
            recommendations.append(f"Zone {zone} has {breakdowns} warehouse breakdowns. Consider preventive maintenance.")
    
    # If no specific recommendations, add general ones
    if not recommendations:
        recommendations = [
            "Consider implementing a just-in-time inventory system to reduce storage costs.",
            "Evaluate transportation routes to minimize delivery time and costs.",
            "Implement regular maintenance schedules for all warehouses to prevent breakdowns.",
            "Consider consolidating warehouses in regions with low utilization."
        ]
    
    return recommendations
