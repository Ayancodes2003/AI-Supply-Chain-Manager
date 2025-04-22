import streamlit as st
import os
import pickle
from pathlib import Path

def setup_auth():
    """
    Set up simplified authentication for the Streamlit app

    Returns:
        tuple: (authenticator, name, authentication_status)
    """
    # Create a simple authenticator class
    class SimpleAuthenticator:
        def __init__(self):
            self.users = {
                'admin': {'password': 'admin', 'name': 'Admin User'},
                'user': {'password': 'password', 'name': 'Regular User'}
            }

        def logout(self, button_name, location):
            if st.button(button_name):
                st.session_state['authentication_status'] = None
                st.session_state['username'] = None
                st.session_state['name'] = None
                st.rerun()

    # Create config directory if it doesn't exist
    config_dir = Path("./config")
    if not config_dir.exists():
        config_dir.mkdir(parents=True, exist_ok=True)

    # Initialize session state
    if 'authentication_status' not in st.session_state:
        st.session_state['authentication_status'] = None
    if 'username' not in st.session_state:
        st.session_state['username'] = None
    if 'name' not in st.session_state:
        st.session_state['name'] = None

    # Create authenticator
    authenticator = SimpleAuthenticator()

    # If not authenticated, show login form
    if st.session_state['authentication_status'] != True:
        st.title('Login')
        username = st.text_input('Username')
        password = st.text_input('Password', type='password')

        if st.button('Login'):
            if username in authenticator.users and authenticator.users[username]['password'] == password:
                st.session_state['authentication_status'] = True
                st.session_state['username'] = username
                st.session_state['name'] = authenticator.users[username]['name']
                st.rerun()
            else:
                st.session_state['authentication_status'] = False
                st.error('Invalid username or password')

    # Return authentication info
    return authenticator, st.session_state.get('name'), st.session_state.get('authentication_status')

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
