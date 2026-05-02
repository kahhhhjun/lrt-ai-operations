# LRT AI Operations - Authentication System

## Overview
This authentication system provides secure access control for the LRT AI Operations decision-support system. Staff must register and login before accessing the main application.

## Features
- **Staff Registration**: New staff can sign up with their name, staff ID, and password
- **Secure Login**: Authentication using staff ID and password
- **Session Management**: Stay logged in during your session
- **User Data Persistence**: All user credentials are stored securely in a file
- **Logout**: Secure logout functionality (data remains saved)

## How to Use

### Starting the Application
Instead of running `streamlit run app.py`, now run:
```bash
streamlit run auth.py
```

### First Time Setup
1. Open the application in your browser
2. You'll see the authentication page with two tabs: "Login" and "Sign Up"

### Registration
1. Click on the "Sign Up" tab
2. Fill in your details:
   - **Full Name**: Your complete name
   - **Staff ID**: Create a unique staff ID (this will be your username)
   - **Password**: Create a secure password (minimum 4 characters)
   - **Confirm Password**: Re-enter your password
3. Click "Sign Up"
4. You'll see a success message if registration is successful

### Login
1. Click on the "Login" tab
2. Enter your Staff ID and Password
3. Click "Login"
4. If successful, you'll be redirected to the main LRT AI Operations application

### Logout
- Click the "🚪 Logout" button in the sidebar to end your session
- Your data will remain saved for next time

## Security Features
- **Password Hashing**: All passwords are securely hashed using SHA-256
- **Staff ID Validation**: Prevents duplicate staff ID registrations
- **Session Management**: Tracks authenticated sessions
- **Data Persistence**: User data is stored in `data/users.json`

## User Data Storage
User credentials are stored in: `data/users.json`

Format:
```json
[
  {
    "name": "Staff Name",
    "staff_id": "STAFF001",
    "password": "hashed_password",
    "created_at": "2026-05-02T10:30:00",
    "last_login": "2026-05-02T14:20:00"
  }
]
```

## Important Notes
- **Original App Unchanged**: Your original `app.py` remains completely unchanged
- **No Data Loss**: Logout doesn't delete any user data or schedules
- **Internal Use**: This is designed for internal staff use only
- **Admin Access**: To manage users, you can directly edit `data/users.json`

## Troubleshooting

### Issues with registration
- Ensure your Staff ID is unique (no duplicates allowed)
- Password must be at least 4 characters long
- All fields are required

### Login problems
- Check that you're using the correct Staff ID
- Verify your password is correct
- If you forget your password, you'll need to contact an administrator to reset it in `data/users.json`

### File permissions
- Make sure the `data/` directory exists
- Ensure the application has write permissions for `data/users.json`

## Migration from Original App
If you were previously using `streamlit run app.py`, simply change your startup command to:
```bash
streamlit run auth.py
```

All existing functionality remains the same - the authentication layer is added on top.