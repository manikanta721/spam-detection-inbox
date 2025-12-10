"""
View all registered users in ChromaDB
Shows username and password hash
"""
import chromadb
import os

# Connect to ChromaDB
vectordb_path = os.path.join(os.path.dirname(__file__), 'chroma_db')
client = chromadb.PersistentClient(path=vectordb_path)
user_collection = client.get_or_create_collection(name="users")

print("=" * 60)
print("USER DATABASE VIEWER")
print("=" * 60)
print(f"\nDatabase Location: {vectordb_path}")
print(f"Collection: users\n")

# Get all users
try:
    results = user_collection.get()
    
    if not results['ids']:
        print("No users found in database.")
        print("Register a user through the website to see data here.")
    else:
        print(f"Total Users: {len(results['ids'])}\n")
        print("-" * 60)
        
        for i, (user_id, metadata) in enumerate(zip(results['ids'], results['metadatas']), 1):
            print(f"\nUser #{i}:")
            print(f"  Username: {user_id}")
            print(f"  Password Hash: {metadata.get('password_hash', 'N/A')}")
            print(f"  Metadata: {metadata}")
        
        print("\n" + "-" * 60)
        print(f"\nTotal: {len(results['ids'])} user(s)")

except Exception as e:
    print(f"Error reading database: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("\nTo add users: Register through the website at http://127.0.0.1:5000/register")
print("=" * 60)

