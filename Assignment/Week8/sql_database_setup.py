import pandas as pd
import sqlite3
import sys

# 1. Load and Clean Data from Excel
print("Loading data from Excel file...")
try:
    df = pd.read_excel("Online Retail.xlsx")
    # Clean the data: remove rows with no CustomerID, no InvoiceNo, and cancelations
    df.dropna(subset=['CustomerID', 'InvoiceNo'], inplace=True)
    df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]
    # Convert CustomerID to integer
    df['CustomerID'] = df['CustomerID'].astype(int)
    # Ensure all StockCodes are strings
    df['StockCode'] = df['StockCode'].astype(str)
    print(f"Data loaded successfully. Found {len(df)} valid records.")
except FileNotFoundError:
    print("Error: 'Online Retail.xlsx' not found. Please download the file.")
    sys.exit()

# 2. Connect to SQLite Database
# This will create a file named 'retail_sql.db' in your directory
DB_NAME = "retail_sql.db"
conn = sqlite3.connect(DB_NAME)
cur = conn.cursor()
print(f"Connected to SQLite database: {DB_NAME}")

# 3. Create Tables (SQL DDL) 
# We drop tables if they exist to make the script re-runnable
print("Creating database tables...")
cur.execute("DROP TABLE IF EXISTS InvoiceItems")
cur.execute("DROP TABLE IF EXISTS Invoices")
cur.execute("DROP TABLE IF EXISTS Products")
cur.execute("DROP TABLE IF EXISTS Customers")

# Customers Table
cur.execute("""
    CREATE TABLE Customers (
        CustomerID INTEGER PRIMARY KEY,
        Country TEXT
    )
""")

# Products Table
cur.execute("""
    CREATE TABLE Products (
        StockCode TEXT PRIMARY KEY,
        Description TEXT,
        UnitPrice REAL
    )
""")

# Invoices Table
cur.execute("""
    CREATE TABLE Invoices (
        InvoiceNo TEXT PRIMARY KEY,
        CustomerID INTEGER,
        InvoiceDate TEXT,
        FOREIGN KEY (CustomerID) REFERENCES Customers(CustomerID)
    )
""")

# InvoiceItems Table (Junction Table)
cur.execute("""
    CREATE TABLE InvoiceItems (
        InvoiceItemID INTEGER PRIMARY KEY AUTOINCREMENT,
        InvoiceNo TEXT,
        StockCode TEXT,
        Quantity INTEGER,
        FOREIGN KEY (InvoiceNo) REFERENCES Invoices(InvoiceNo),
        FOREIGN KEY (StockCode) REFERENCES Products(StockCode)
    )
""")
print("Tables created successfully.")
conn.commit()

# 4. Prepare and Insert Data
# Get the first 1000 unique transactions for insertion
df_subset = df.head(1000)

# Use sets to keep track of what we've already inserted to avoid errors
inserted_customers = set()
inserted_products = set()
inserted_invoices = set()

print("Inserting 1000 records into the database...")
# Use a try-except block to handle any potential data errors gracefully
try:
    for index, row in df_subset.iterrows():
        # Insert into Customers (if not already present)
        if row['CustomerID'] not in inserted_customers:
            cur.execute("INSERT INTO Customers (CustomerID, Country) VALUES (?, ?)", 
                        (row['CustomerID'], row['Country']))
            inserted_customers.add(row['CustomerID'])

        # Insert into Products (if not already present)
        if row['StockCode'] not in inserted_products:
            cur.execute("INSERT INTO Products (StockCode, Description, UnitPrice) VALUES (?, ?, ?)",
                        (row['StockCode'], row['Description'], row['UnitPrice']))
            inserted_products.add(row['StockCode'])

        # Insert into Invoices (if not already present)
        if row['InvoiceNo'] not in inserted_invoices:
            cur.execute("INSERT INTO Invoices (InvoiceNo, CustomerID, InvoiceDate) VALUES (?, ?, ?)",
                        (row['InvoiceNo'], row['CustomerID'], str(row['InvoiceDate'])))
            inserted_invoices.add(row['InvoiceNo'])

        # Insert into InvoiceItems (the main transaction line)
        cur.execute("INSERT INTO InvoiceItems (InvoiceNo, StockCode, Quantity) VALUES (?, ?, ?)",
                    (row['InvoiceNo'], row['StockCode'], row['Quantity']))

    conn.commit()
    print("Data insertion complete.")

except sqlite3.IntegrityError as e:
    print(f"An integrity error occurred: {e}")
    print("Rolling back transaction.")
    conn.rollback()
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    print("Rolling back transaction.")
    conn.rollback()

finally:
    # 5. Close the Connection 
    conn.close()
    print("SQLite connection closed.")
