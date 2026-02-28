"""
Student Database Management Tool
Easily add/edit student information
"""

import json
import os

def load_database():
    """Load student database"""
    db_path = "students_database.json"
    if os.path.exists(db_path):
        with open(db_path, 'r') as f:
            return json.load(f)
    else:
        return {"students": {}}

def save_database(data):
    """Save student database"""
    with open("students_database.json", 'w') as f:
        json.dump(data, f, indent=2)
    print("✅ Database saved successfully!")

def add_student():
    """Add new student to database"""
    print("\n📝 Add New Student")
    print("="*50)
    
    folder_name = input("Enter folder name: ").strip()
    if not folder_name:
        print("❌ Folder name cannot be empty!")
        return
    
    full_name = input("Enter full name: ").strip()
    reg_number = input("Enter registration number: ").strip()
    course = input("Enter course: ").strip()
    year = input("Enter year: ").strip()
    email = input("Enter email: ").strip()
    
    data = load_database()
    
    data['students'][folder_name] = {
        'full_name': full_name,
        'registration_number': reg_number,
        'course': course,
        'year': year,
        'email': email
    }
    
    save_database(data)
    print(f"\n✅ Student '{full_name}' added successfully!")

def edit_student():
    """Edit existing student"""
    data = load_database()
    
    print("\n📝 Edit Student")
    print("="*50)
    print("\nExisting students:")
    for i, name in enumerate(data['students'].keys(), 1):
        print(f"{i}. {name}")
    
    folder_name = input("\nEnter folder name to edit: ").strip()
    
    if folder_name not in data['students']:
        print(f"❌ Student '{folder_name}' not found!")
        return
    
    student = data['students'][folder_name]
    
    print(f"\nCurrent information for {folder_name}:")
    print(f"Full Name: {student['full_name']}")
    print(f"Registration Number: {student['registration_number']}")
    print(f"Course: {student['course']}")
    print(f"Year: {student['year']}")
    print(f"Email: {student['email']}")
    
    print("\nEnter new values (press Enter to keep current):")
    
    full_name = input(f"Full name [{student['full_name']}]: ").strip()
    reg_number = input(f"Registration number [{student['registration_number']}]: ").strip()
    course = input(f"Course [{student['course']}]: ").strip()
    year = input(f"Year [{student['year']}]: ").strip()
    email = input(f"Email [{student['email']}]: ").strip()
    
    # Update only if new value provided
    if full_name:
        student['full_name'] = full_name
    if reg_number:
        student['registration_number'] = reg_number
    if course:
        student['course'] = course
    if year:
        student['year'] = year
    if email:
        student['email'] = email
    
    save_database(data)
    print(f"\n✅ Student '{folder_name}' updated successfully!")

def view_students():
    """View all students"""
    data = load_database()
    
    print("\n👥 All Students")
    print("="*80)
    
    if not data['students']:
        print("No students in database.")
        return
    
    print(f"{'Folder Name':<15} {'Full Name':<25} {'Reg No':<12} {'Course':<20}")
    print("-"*80)
    
    for folder_name, info in data['students'].items():
        print(f"{folder_name:<15} {info['full_name']:<25} {info['registration_number']:<12} {info['course']:<20}")
    
    print(f"\nTotal students: {len(data['students'])}")

def delete_student():
    """Delete student from database"""
    data = load_database()
    
    print("\n🗑️  Delete Student")
    print("="*50)
    
    view_students()
    
    folder_name = input("\nEnter folder name to delete: ").strip()
    
    if folder_name not in data['students']:
        print(f"❌ Student '{folder_name}' not found!")
        return
    
    confirm = input(f"Are you sure you want to delete '{folder_name}'? (yes/no): ").strip().lower()
    
    if confirm == 'yes':
        del data['students'][folder_name]
        save_database(data)
        print(f"✅ Student '{folder_name}' deleted successfully!")
    else:
        print("❌ Deletion cancelled.")

def import_from_folders():
    """Auto-import students from dataset_cropped folders"""
    print("\n📂 Import from Dataset Folders")
    print("="*50)
    
    dataset_path = "dataset_cropped"
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset folder '{dataset_path}' not found!")
        return
    
    folders = [f for f in os.listdir(dataset_path) 
              if os.path.isdir(os.path.join(dataset_path, f)) and not f.startswith('_')]
    
    if not folders:
        print("No student folders found!")
        return
    
    data = load_database()
    added = 0
    
    for folder_name in folders:
        if folder_name not in data['students']:
            print(f"\nFound new folder: {folder_name}")
            full_name = input(f"  Enter full name (or press Enter to use '{folder_name}'): ").strip()
            if not full_name:
                full_name = folder_name
            
            reg_number = input(f"  Enter registration number: ").strip()
            course = input(f"  Enter course: ").strip()
            year = input(f"  Enter year: ").strip()
            email = input(f"  Enter email: ").strip()
            
            data['students'][folder_name] = {
                'full_name': full_name,
                'registration_number': reg_number or 'N/A',
                'course': course or 'N/A',
                'year': year or 'N/A',
                'email': email or 'N/A'
            }
            added += 1
    
    if added > 0:
        save_database(data)
        print(f"\n✅ Added {added} new student(s)!")
    else:
        print("\n✅ All students already in database!")

def main():
    print("🎓 Student Database Management")
    print("="*50)
    
    while True:
        print("\nOptions:")
        print("1. View all students")
        print("2. Add new student")
        print("3. Edit student")
        print("4. Delete student")
        print("5. Import from dataset folders")
        print("6. Exit")
        
        choice = input("\nEnter choice (1-6): ").strip()
        
        if choice == '1':
            view_students()
        elif choice == '2':
            add_student()
        elif choice == '3':
            edit_student()
        elif choice == '4':
            delete_student()
        elif choice == '5':
            import_from_folders()
        elif choice == '6':
            print("\n👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice!")

if __name__ == "__main__":
    main()
