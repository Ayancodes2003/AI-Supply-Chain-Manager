import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, plot_tree, export_text
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import io
import base64
from PIL import Image

def perform_regression_analysis(df, target_col, feature_cols):
    """
    Perform regression analysis on the data
    
    Args:
        df (pandas.DataFrame): Data to analyze
        target_col (str): Target column name
        feature_cols (list): List of feature column names
        
    Returns:
        dict: Dictionary containing regression results
    """
    # Check if required columns exist
    if target_col not in df.columns or not all(col in df.columns for col in feature_cols):
        return {
            'success': False,
            'message': 'One or more selected columns do not exist in the dataset'
        }
    
    # Remove rows with missing values
    data = df[[target_col] + feature_cols].dropna()
    
    if len(data) < 10:
        return {
            'success': False,
            'message': 'Not enough data points for regression analysis'
        }
    
    # Split features and target
    X = data[feature_cols]
    y = data[target_col]
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Calculate metrics
    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    
    # Get coefficients
    coefficients = pd.DataFrame({
        'Feature': feature_cols,
        'Coefficient': model.coef_
    })
    
    # Create scatter plot of actual vs predicted values
    fig = px.scatter(
        x=y_test, 
        y=y_pred_test,
        labels={'x': f'Actual {target_col}', 'y': f'Predicted {target_col}'},
        title='Actual vs Predicted Values'
    )
    
    # Add 45-degree line
    fig.add_trace(
        go.Scatter(
            x=[y_test.min(), y_test.max()],
            y=[y_test.min(), y_test.max()],
            mode='lines',
            line=dict(color='red', dash='dash'),
            name='Perfect Prediction'
        )
    )
    
    return {
        'success': True,
        'model': model,
        'coefficients': coefficients,
        'metrics': {
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'intercept': model.intercept_
        },
        'plot': fig
    }

def perform_decision_tree_analysis(df, target_col, feature_cols, max_depth=3):
    """
    Perform decision tree analysis on the data
    
    Args:
        df (pandas.DataFrame): Data to analyze
        target_col (str): Target column name
        feature_cols (list): List of feature column names
        max_depth (int): Maximum depth of the decision tree
        
    Returns:
        dict: Dictionary containing decision tree results
    """
    # Check if required columns exist
    if target_col not in df.columns or not all(col in df.columns for col in feature_cols):
        return {
            'success': False,
            'message': 'One or more selected columns do not exist in the dataset'
        }
    
    # Remove rows with missing values
    data = df[[target_col] + feature_cols].dropna()
    
    if len(data) < 10:
        return {
            'success': False,
            'message': 'Not enough data points for decision tree analysis'
        }
    
    # Split features and target
    X = data[feature_cols]
    y = data[target_col]
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train decision tree model
    model = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Calculate metrics
    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    
    # Get feature importances
    importances = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    # Create decision tree visualization
    plt.figure(figsize=(12, 8))
    plot_tree(
        model, 
        feature_names=feature_cols,
        filled=True, 
        rounded=True, 
        fontsize=10
    )
    
    # Convert plot to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    img = Image.open(buf)
    
    # Convert image to base64 string
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    # Get tree rules as text
    tree_rules = export_text(model, feature_names=feature_cols)
    
    return {
        'success': True,
        'model': model,
        'feature_importance': importances,
        'metrics': {
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_r2': train_r2,
            'test_r2': test_r2
        },
        'tree_image': img_str,
        'tree_rules': tree_rules
    }

def perform_clustering_analysis(df, feature_cols, n_clusters=3):
    """
    Perform clustering analysis on the data
    
    Args:
        df (pandas.DataFrame): Data to analyze
        feature_cols (list): List of feature column names
        n_clusters (int): Number of clusters
        
    Returns:
        dict: Dictionary containing clustering results
    """
    # Check if required columns exist
    if not all(col in df.columns for col in feature_cols):
        return {
            'success': False,
            'message': 'One or more selected columns do not exist in the dataset'
        }
    
    # Remove rows with missing values
    data = df[feature_cols].dropna()
    
    if len(data) < 10:
        return {
            'success': False,
            'message': 'Not enough data points for clustering analysis'
        }
    
    # Scale the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(scaled_data)
    
    # Add cluster labels to the original data
    result_df = data.copy()
    result_df['Cluster'] = clusters
    
    # Calculate cluster centers
    centers = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=feature_cols)
    centers['Cluster'] = range(n_clusters)
    
    # Create cluster visualization (first two features)
    if len(feature_cols) >= 2:
        fig = px.scatter(
            result_df,
            x=feature_cols[0],
            y=feature_cols[1],
            color='Cluster',
            title=f'Clustering by {feature_cols[0]} and {feature_cols[1]}',
            color_continuous_scale=px.colors.qualitative.G10
        )
        
        # Add cluster centers
        fig.add_trace(
            go.Scatter(
                x=centers[feature_cols[0]],
                y=centers[feature_cols[1]],
                mode='markers',
                marker=dict(
                    color='black',
                    size=12,
                    symbol='x'
                ),
                name='Cluster Centers'
            )
        )
    else:
        fig = None
    
    # Calculate cluster statistics
    cluster_stats = result_df.groupby('Cluster').agg(['mean', 'std', 'count'])
    
    return {
        'success': True,
        'model': kmeans,
        'clustered_data': result_df,
        'cluster_centers': centers,
        'cluster_stats': cluster_stats,
        'plot': fig
    }

def perform_what_if_analysis(df, model, feature_cols, target_col, what_if_values):
    """
    Perform what-if analysis using a trained model
    
    Args:
        df (pandas.DataFrame): Original data
        model: Trained model
        feature_cols (list): List of feature column names
        target_col (str): Target column name
        what_if_values (dict): Dictionary of feature values to use for prediction
        
    Returns:
        dict: Dictionary containing what-if analysis results
    """
    # Check if all required features are provided
    if not all(col in what_if_values for col in feature_cols):
        return {
            'success': False,
            'message': 'Values for all features must be provided'
        }
    
    # Create a DataFrame with the what-if values
    what_if_df = pd.DataFrame([what_if_values])
    
    # Make prediction
    prediction = model.predict(what_if_df[feature_cols])[0]
    
    # Get feature statistics from original data
    feature_stats = df[feature_cols].describe()
    
    # Compare what-if values with feature statistics
    comparison = pd.DataFrame({
        'Feature': feature_cols,
        'What-If Value': [what_if_values[col] for col in feature_cols],
        'Original Mean': feature_stats.loc['mean'].values,
        'Original Min': feature_stats.loc['min'].values,
        'Original Max': feature_stats.loc['max'].values
    })
    
    # Calculate percentile of each what-if value
    for i, col in enumerate(feature_cols):
        value = what_if_values[col]
        percentile = (df[col] < value).mean() * 100
        comparison.loc[i, 'Percentile'] = percentile
    
    return {
        'success': True,
        'prediction': prediction,
        'comparison': comparison,
        'target_col': target_col
    }
