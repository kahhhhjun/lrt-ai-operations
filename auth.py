"""Authentication system for LRT AI Operations.
Run: streamlit run auth.py"""

import json
import hashlib
import os
from pathlib import Path
from datetime import datetime
import streamlit as st

# Configuration
USERS_FILE = Path(__file__).parent / "data" / "users.json"
SESSION_KEY = "authenticated"
USER_KEY = "current_user"

def init_users_file():
    """Initialize the users file if it doesn't exist."""
    USERS_FILE.parent.mkdir(exist_ok=True)
    if not USERS_FILE.exists():
        with open(USERS_FILE, 'w') as f:
            json.dump([], f)
        return True
    return False

def hash_password(password):
    """Hash password using SHA256."""
    return hashlib.sha256(password.encode()).hexdigest()

def load_users():
    """Load users from file."""
    try:
        with open(USERS_FILE, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []

def save_users(users):
    """Save users to file."""
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f, indent=2)

def register_user(name, staff_id, password):
    """Register a new user."""
    users = load_users()

    # Check if staff_id already exists
    for user in users:
        if user['staff_id'] == staff_id:
            return False, "Staff ID already registered"

    # Create new user
    new_user = {
        'name': name,
        'staff_id': staff_id,
        'password': hash_password(password),
        'created_at': datetime.now().isoformat(),
        'last_login': None
    }

    users.append(new_user)
    save_users(users)
    return True, "Registration successful"

def authenticate_user(staff_id, password):
    """Authenticate user with staff ID and password."""
    users = load_users()

    for user in users:
        if user['staff_id'] == staff_id and user['password'] == hash_password(password):
            # Update last login
            user['last_login'] = datetime.now().isoformat()
            save_users(users)
            return True, user
        elif user['staff_id'] == staff_id and user['password'] != hash_password(password):
            return False, "Incorrect password"

    return False, "Staff ID not found"

def logout_user():
    """Logout current user."""
    # Only delete authentication-related keys, not Streamlit's internal state
    keys_to_delete = [SESSION_KEY, USER_KEY]
    for key in keys_to_delete:
        if key in st.session_state:
            del st.session_state[key]

def is_authenticated():
    """Check if user is authenticated."""
    return st.session_state.get(SESSION_KEY, False)

def get_current_user():
    """Get current authenticated user."""
    return st.session_state.get(USER_KEY)

# ── UI Components ────────────────────────────────────────────────────────────

def show_login_page():
    """Display login page."""
    st.set_page_config(page_title="Login - LRT AI Operations", layout="centered")

    st.title("🔐 LRT AI Operations")
    st.caption("Staff Authentication System")

    tab1, tab2 = st.tabs(["Login", "Sign Up"])

    # Login Tab
    with tab1:
        st.subheader("Staff Login")
        login_staff_id = st.text_input("Staff ID", placeholder="Enter your staff ID", key="login_staff_id")
        login_password = st.text_input("Password", type="password", placeholder="Enter your password", key="login_password")

        if st.button("Login", type="primary", use_container_width=True):
            if not login_staff_id or not login_password:
                st.error("Please enter both Staff ID and password")
            else:
                success, result = authenticate_user(login_staff_id, login_password)
                if success:
                    st.session_state[SESSION_KEY] = True
                    st.session_state[USER_KEY] = {
                        'name': result['name'],
                        'staff_id': result['staff_id']
                    }
                    st.success(f"Welcome back, {result['name']}!")
                    st.rerun()
                else:
                    st.error(result)

    # Sign Up Tab
    with tab2:
        st.subheader("Staff Registration")
        reg_name = st.text_input("Full Name", placeholder="Enter your full name", key="reg_name")
        reg_staff_id = st.text_input("Staff ID", placeholder="Create your staff ID", key="reg_staff_id")
        reg_password = st.text_input("Password", type="password", placeholder="Create a password", key="reg_password")
        reg_confirm_password = st.text_input("Confirm Password", type="password", placeholder="Confirm your password", key="reg_confirm_password")

        if st.button("Sign Up", type="primary", use_container_width=True):
            if not all([reg_name, reg_staff_id, reg_password, reg_confirm_password]):
                st.error("Please fill in all fields")
            elif reg_password != reg_confirm_password:
                st.error("Passwords do not match")
            elif len(reg_password) < 4:
                st.error("Password must be at least 4 characters long")
            else:
                success, message = register_user(reg_name, reg_staff_id, reg_password)
                if success:
                    st.success(message)
                    st.info("Please go to Login tab to access the system")
                else:
                    st.error(message)

    # Footer
    st.markdown("---")
    st.caption("💡 New staff? Please register first using your name, staff ID, and create a password.")
    st.caption("🔒 Your credentials are stored securely and only used for authentication.")

def show_logout_button():
    """Display logout button in sidebar."""
    with st.sidebar:
        st.markdown("---")
        if st.button("🚪 Logout", use_container_width=True):
            logout_user()
            st.rerun()

        # Show user info
        user = get_current_user()
        if user:
            st.info(f"👤 Logged in as: **{user['name']}**\n\n🆔 Staff ID: {user['staff_id']}")

def main():
    """Main authentication flow."""
    # Initialize users file
    init_users_file()

    # Check authentication
    if is_authenticated():
        # Show logout button FIRST, before importing app
        # This ensures it stays visible even when app.py calls st.rerun()
        show_logout_button()

        # User is authenticated, show the main app
        # Force fresh import by removing from cache
        import sys
        if 'app' in sys.modules:
            del sys.modules['app']
        # Import app module - this will run fresh
        import app
    else:
        # Show login/signup page
        show_login_page()

if __name__ == "__main__":
    main()