import pandas as pd
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure

# 1. Load and Clean Data
print("Loading data from Excel file...")
try:
    df = pd.read_excel("Online Retail.xlsx")
    df.dropna(subset=['CustomerID'], inplace=True)
    df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]
    df['CustomerID'] = df['CustomerID'].astype(int)
    # Limit to the first 1000 transactions for processing
    df = df.head(1000)
    print(f"Data loaded successfully. Processing {len(df)} records.")
except FileNotFoundError:
    print("Error: 'Online Retail.xlsx' not found. Please download the file.")
    exit()

# 2. Connect to MongoDB (Using a new Database name)
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "retail_db_customer"
COLLECTION_NAME = "customers"

print("\nConnecting to MongoDB...")
try:
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    client.admin.command('ping')
    print("MongoDB connection successful.")
except ConnectionFailure as e:
    print(f"Could not connect to MongoDB: {e}")
    exit()

db = client[DB_NAME]
collection = db[COLLECTION_NAME]

print(f"Clearing old data from '{COLLECTION_NAME}' collection...")
collection.delete_many({})

# 3. Process Data into Customer-Centric Format
print("Processing data into customer-centric format...")
customers = {}
for index, row in df.iterrows():
    customer_id = row['CustomerID']
    # If this is the first time we see this customer, create a base document
    if customer_id not in customers:
        customers[customer_id] = {
            '_id': customer_id,
            'Country': row['Country'],
            'invoices': []
        }
    # Append the transaction details to the 'invoices' array
    customers[customer_id]['invoices'].append({
        'InvoiceNo': row['InvoiceNo'],
        'InvoiceDate': row['InvoiceDate'],
        'StockCode': row['StockCode'],
        'Description': row['Description'],
        'Quantity': row['Quantity'],
        'UnitPrice': row['UnitPrice']
    })

# Convert the dictionary of customers to a list for insertion
data_to_insert = list(customers.values())

print(f"Inserting {len(data_to_insert)} customer documents into MongoDB...")
try:
    collection.insert_many(data_to_insert)
    print("Data insertion complete.")
except Exception as e:
    print(f"An error occurred during data insertion: {e}")

# 4. Close the connection
client.close()
print("\nMongoDB connection closed.")
