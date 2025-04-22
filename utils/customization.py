import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

def customize_dashboard():
    """
    Allow users to customize the dashboard appearance
    
    Returns:
        dict: Customization settings
    """
    st.sidebar.markdown("## Dashboard Customization")
    
    # Theme selection
    theme = st.sidebar.selectbox(
        "Theme",
        options=["Light", "Dark", "Blue", "Green"],
        index=0
    )
    
    # Chart type preference
    chart_type = st.sidebar.selectbox(
        "Default Chart Type",
        options=["Bar", "Line", "Scatter", "Pie"],
        index=0
    )
    
    # Layout options
    layout = st.sidebar.selectbox(
        "Layout",
        options=["Standard", "Compact", "Expanded"],
        index=0
    )
    
    # Data preview toggle
    show_data_preview = st.sidebar.checkbox("Show Data Preview", value=True)
    
    # Default view
    default_view = st.sidebar.selectbox(
        "Default View",
        options=["Dashboard", "Analytics", "Data Explorer"],
        index=0
    )
    
    # Save button
    if st.sidebar.button("Save Preferences"):
        st.sidebar.success("Preferences saved!")
    
    # Return settings
    return {
        "theme": theme,
        "chart_type": chart_type,
        "layout": layout,
        "show_data_preview": show_data_preview,
        "default_view": default_view
    }

def apply_theme(theme):
    """
    Apply the selected theme to the dashboard
    
    Args:
        theme (str): Selected theme
    """
    if theme == "Light":
        # Light theme (default)
        st.markdown("""
        <style>
            .main-header {color: #1E88E5;}
            .sub-header {color: #424242;}
            .kpi-card {background-color: #f5f5f5;}
            .kpi-value {color: #1E88E5;}
            .kpi-label {color: #616161;}
        </style>
        """, unsafe_allow_html=True)
    
    elif theme == "Dark":
        # Dark theme
        st.markdown("""
        <style>
            .main-header {color: #90CAF9;}
            .sub-header {color: #E0E0E0;}
            .kpi-card {background-color: #333333;}
            .kpi-value {color: #90CAF9;}
            .kpi-label {color: #BDBDBD;}
            
            /* Override Streamlit's default styles */
            .stApp {
                background-color: #121212;
                color: #E0E0E0;
            }
            .stDataFrame {
                background-color: #333333;
            }
            .stSelectbox label, .stMultiSelect label {
                color: #E0E0E0 !important;
            }
        </style>
        """, unsafe_allow_html=True)
    
    elif theme == "Blue":
        # Blue theme
        st.markdown("""
        <style>
            .main-header {color: #1565C0;}
            .sub-header {color: #0D47A1;}
            .kpi-card {background-color: #E3F2FD;}
            .kpi-value {color: #1565C0;}
            .kpi-label {color: #1976D2;}
            
            /* Override Streamlit's default styles */
            .stApp {
                background-color: #BBDEFB;
                color: #0D47A1;
            }
        </style>
        """, unsafe_allow_html=True)
    
    elif theme == "Green":
        # Green theme
        st.markdown("""
        <style>
            .main-header {color: #2E7D32;}
            .sub-header {color: #1B5E20;}
            .kpi-card {background-color: #E8F5E9;}
            .kpi-value {color: #2E7D32;}
            .kpi-label {color: #388E3C;}
            
            /* Override Streamlit's default styles */
            .stApp {
                background-color: #C8E6C9;
                color: #1B5E20;
            }
        </style>
        """, unsafe_allow_html=True)

def get_chart_template(theme):
    """
    Get the appropriate Plotly template based on the selected theme
    
    Args:
        theme (str): Selected theme
        
    Returns:
        str: Plotly template name
    """
    theme_templates = {
        "Light": "plotly_white",
        "Dark": "plotly_dark",
        "Blue": "plotly",
        "Green": "plotly_white"
    }
    
    return theme_templates.get(theme, "plotly_white")

def create_chart(data, chart_type, x, y, color=None, title=None, template="plotly_white"):
    """
    Create a chart based on the selected type and data
    
    Args:
        data (pandas.DataFrame): Data for the chart
        chart_type (str): Type of chart to create
        x (str): Column name for x-axis
        y (str or list): Column name(s) for y-axis
        color (str, optional): Column name for color
        title (str, optional): Chart title
        template (str, optional): Plotly template
        
    Returns:
        plotly.graph_objects.Figure: Created chart
    """
    if chart_type == "Bar":
        fig = px.bar(data, x=x, y=y, color=color, title=title, template=template)
    
    elif chart_type == "Line":
        fig = px.line(data, x=x, y=y, color=color, title=title, template=template)
    
    elif chart_type == "Scatter":
        fig = px.scatter(data, x=x, y=y, color=color, title=title, template=template)
    
    elif chart_type == "Pie":
        fig = px.pie(data, values=y, names=x, title=title, template=template)
    
    else:
        # Default to bar chart
        fig = px.bar(data, x=x, y=y, color=color, title=title, template=template)
    
    return fig
