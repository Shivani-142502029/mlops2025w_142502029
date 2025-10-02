import pandas as pd
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure

# 1. Load and Clean Data
print("Loading data from Excel file...")
try:
    df = pd.read_excel("Online Retail.xlsx")
    # Clean the data: remove rows with no CustomerID and cancelations
    df.dropna(subset=['CustomerID'], inplace=True)
    df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]
    # Convert CustomerID to integer
    df['CustomerID'] = df['CustomerID'].astype(int)
    print(f"Data loaded successfully. Found {len(df)} valid records.")
except FileNotFoundError:
    print("Error: 'Online Retail.xlsx' not found. Please download the file.")
    exit()

# --- 2. Connect to MongoDB with Error Handling ---
# This is the connection string for your Docker container.
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "retail_db_transaction"
COLLECTION_NAME = "invoices"

print("\nConnecting to MongoDB...")
try:
    # MongoClient manages connection pooling automatically.
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    # The ismaster command is cheap and does not require auth.
    client.admin.command('ping')
    print("MongoDB connection successful.")
except ConnectionFailure as e:
    print(f"Could not connect to MongoDB: {e}")
    exit()

# --- 3. Insert Data into MongoDB ---
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

# Clear the collection to avoid duplicate data on re-runs
print(f"Clearing old data from '{COLLECTION_NAME}' collection...")
collection.delete_many({})

# Convert the DataFrame to a list of dictionaries (JSON-like format)
data_to_insert = df.to_dict(orient='records')

# We will insert a limited number of records as per the assignment
records_to_insert = data_to_insert[:1000]

print(f"Inserting {len(records_to_insert)} records into MongoDB...")
try:
    collection.insert_many(records_to_insert)
    print("Data insertion complete.")
except Exception as e:
    print(f"An error occurred during data insertion: {e}")

# --- 4. Close the connection ---
client.close()
print("\nMongoDB connection closed.")
