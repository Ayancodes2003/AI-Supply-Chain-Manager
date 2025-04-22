import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
import os
import pickle
from pathlib import Path

def setup_auth():
    """
    Set up authentication for the Streamlit app
    
    Returns:
        tuple: (authenticator, name, authentication_status)
    """
    # Define the config file path
    config_dir = Path("./config")
    config_file = config_dir / "auth.yaml"
    
    # Create config directory if it doesn't exist
    if not config_dir.exists():
        config_dir.mkdir(parents=True, exist_ok=True)
    
    # Create default config if it doesn't exist
    if not config_file.exists():
        default_config = {
            'credentials': {
                'usernames': {
                    'admin': {
                        'email': 'admin@example.com',
                        'name': 'Admin User',
                        'password': stauth.Hasher(['admin']).generate()[0]
                    },
                    'user': {
                        'email': 'user@example.com',
                        'name': 'Regular User',
                        'password': stauth.Hasher(['password']).generate()[0]
                    }
                }
            },
            'cookie': {
                'expiry_days': 30,
                'key': 'fmcg_dashboard_auth',
                'name': 'fmcg_dashboard_auth'
            },
            'preauthorized': {
                'emails': ['admin@example.com']
            }
        }
        
        with open(config_file, 'w') as file:
            yaml.dump(default_config, file, default_flow_style=False)
    
    # Load config
    with open(config_file, 'r') as file:
        config = yaml.load(file, Loader=SafeLoader)
    
    # Create authenticator
    authenticator = stauth.Authenticate(
        config['credentials'],
        config['cookie']['name'],
        config['cookie']['key'],
        config['cookie']['expiry_days'],
        config['preauthorized']
    )
    
    # Authenticate user
    name, authentication_status, username = authenticator.login('Login', 'main')
    
    return authenticator, name, authentication_status

def check_auth():
    """
    Check if user is authenticated
    
    Returns:
        bool: True if authenticated, False otherwise
    """
    if 'authentication_status' not in st.session_state:
        return False
    
    return st.session_state['authentication_status']

def get_user_role():
    """
    Get the role of the authenticated user
    
    Returns:
        str: 'admin' or 'user'
    """
    if 'username' not in st.session_state:
        return None
    
    username = st.session_state['username']
    
    if username == 'admin':
        return 'admin'
    else:
        return 'user'

def save_user_preferences(username, preferences):
    """
    Save user preferences to a file
    
    Args:
        username (str): Username
        preferences (dict): User preferences
    """
    # Create preferences directory if it doesn't exist
    prefs_dir = Path("./config/preferences")
    prefs_dir.mkdir(parents=True, exist_ok=True)
    
    # Save preferences to file
    prefs_file = prefs_dir / f"{username}.pkl"
    with open(prefs_file, 'wb') as file:
        pickle.dump(preferences, file)

def load_user_preferences(username):
    """
    Load user preferences from a file
    
    Args:
        username (str): Username
        
    Returns:
        dict: User preferences
    """
    # Check if preferences file exists
    prefs_file = Path(f"./config/preferences/{username}.pkl")
    
    if prefs_file.exists():
        with open(prefs_file, 'rb') as file:
            return pickle.load(file)
    
    # Return default preferences
    return {
        'theme': 'Light',
        'default_view': 'Dashboard',
        'show_data_preview': True,
        'chart_type': 'Bar'
    }
