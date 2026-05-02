"""Admin utilities for managing users in the authentication system.
Run: python admin_utils.py"""

import json
import hashlib
from pathlib import Path
from datetime import datetime

USERS_FILE = Path(__file__).parent / "data" / "users.json"

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

def list_all_users():
    """List all registered users."""
    users = load_users()
    if not users:
        print("No users registered yet.")
        return

    print("\n" + "="*60)
    print("REGISTERED USERS")
    print("="*60)
    for idx, user in enumerate(users, 1):
        print(f"\n{idx}. {user['name']}")
        print(f"   Staff ID: {user['staff_id']}")
        print(f"   Created: {user['created_at']}")
        print(f"   Last Login: {user['last_login'] or 'Never'}")
    print("\n" + "="*60)

def reset_password(staff_id, new_password):
    """Reset password for a user."""
    users = load_users()
    for user in users:
        if user['staff_id'] == staff_id:
            user['password'] = hash_password(new_password)
            save_users(users)
            print(f"Password reset successfully for {user['name']} ({staff_id})")
            return True
    print(f"User with Staff ID '{staff_id}' not found.")
    return False

def delete_user(staff_id):
    """Delete a user."""
    users = load_users()
    for idx, user in enumerate(users):
        if user['staff_id'] == staff_id:
            deleted_user = users.pop(idx)
            save_users(users)
            print(f"User deleted: {deleted_user['name']} ({staff_id})")
            return True
    print(f"User with Staff ID '{staff_id}' not found.")
    return False

def main():
    """Admin menu."""
    print("\n" + "="*60)
    print("LRT AI OPERATIONS - USER MANAGEMENT")
    print("="*60)

    while True:
        print("\nOptions:")
        print("1. List all users")
        print("2. Reset user password")
        print("3. Delete user")
        print("4. Exit")

        choice = input("\nEnter your choice (1-4): ").strip()

        if choice == "1":
            list_all_users()
        elif choice == "2":
            staff_id = input("Enter Staff ID: ").strip()
            new_password = input("Enter new password: ").strip()
            if len(new_password) < 4:
                print("Password must be at least 4 characters long.")
            else:
                reset_password(staff_id, new_password)
        elif choice == "3":
            staff_id = input("Enter Staff ID to delete: ").strip()
            confirm = input(f"Are you sure you want to delete user '{staff_id}'? (yes/no): ").strip().lower()
            if confirm == "yes":
                delete_user(staff_id)
            else:
                print("Deletion cancelled.")
        elif choice == "4":
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please enter 1-4.")

if __name__ == "__main__":
    main()