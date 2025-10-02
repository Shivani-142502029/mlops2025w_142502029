import time
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure

# Connection Setup
MONGO_URI = "mongodb://localhost:27017/"
client = MongoClient(MONGO_URI)

# Databases and Collections
db_trans = client['retail_db_transaction']
col_trans = db_trans['invoices']

db_cust = client['retail_db_customer']
col_cust = db_cust['customers']

# CRUD Operations and Timing 

# READ TEST
customer_id_to_find = 17850
print(f"\n--- READ TEST: Find all invoices for Customer ID {customer_id_to_find} ---")

# A. Read from Transaction-Centric Model
start_time = time.time()
trans_results = list(col_trans.find({'CustomerID': customer_id_to_find}))
end_time = time.time()
print(f"Transaction-Centric: Found {len(trans_results)} invoices in {end_time - start_time:.6f} seconds.")

# B. Read from Customer-Centric Model
start_time = time.time()
cust_result = col_cust.find_one({'_id': customer_id_to_find})
end_time = time.time()

# We check if the customer was found before counting their invoices
num_invoices = len(cust_result['invoices']) if cust_result else 0
print(f"Customer-Centric:    Found {num_invoices} invoices in {end_time - start_time:.6f} seconds.")

# --- UPDATE TEST ---
print("\n--- UPDATE TEST: Change a product description ---")

# A. Update in Transaction-Centric Model
invoice_to_update = '536365'
stock_code_to_update = '85123A'
start_time = time.time()
col_trans.update_one(
    {'InvoiceNo': invoice_to_update, 'StockCode': stock_code_to_update},
    {'$set': {'Description': 'UPDATED DESCRIPTION'}}
)
end_time = time.time()
print(f"Transaction-Centric: Update took {end_time - start_time:.6f} seconds.")


# B. Update in Customer-Centric Model
start_time = time.time()
col_cust.update_one(
    {'_id': 17850, 'invoices.InvoiceNo': invoice_to_update, 'invoices.StockCode': stock_code_to_update},
    {'$set': {'invoices.$.Description': 'UPDATED DESCRIPTION'}}
)
end_time = time.time()
print(f"Customer-Centric:    Update took {end_time - start_time:.6f} seconds.")

# CREATE TEST
print("\n--- CREATE TEST: Insert a new invoice ---")
new_invoice = {
    'InvoiceNo': '999999', 'StockCode': '12345', 'Description': 'TEST PRODUCT',
    'Quantity': 1, 'InvoiceDate': '2025-10-02 12:00:00', 'UnitPrice': 9.99,
    'CustomerID': 99999, 'Country': 'Testland'
}

# A. Insert into Transaction-Centric Model
start_time = time.time()
col_trans.insert_one(new_invoice)
end_time = time.time()
print(f"Transaction-Centric: Insert took {end_time - start_time:.6f} seconds.")

# B. Insert into Customer-Centric Model (more complex)
# This requires adding an invoice to a customer's array (an "upsert")
start_time = time.time()
col_cust.update_one(
    {'_id': 99999},
    {
        '$push': {'invoices': new_invoice},
        '$setOnInsert': {'Country': 'Testland'}
    },
    upsert=True
)
end_time = time.time()
print(f"Customer-Centric:    Insert (upsert) took {end_time - start_time:.6f} seconds.")

# DELETE TEST
print("\n--- DELETE TEST: Remove the new invoice ---")

# A. Delete from Transaction-Centric Model
start_time = time.time()
col_trans.delete_one({'InvoiceNo': '999999'})
end_time = time.time()
print(f"Transaction-Centric: Delete took {end_time - start_time:.6f} seconds.")

# B. Delete from Customer-Centric Model (also complex)
# This requires "pulling" the invoice from the customer's array
start_time = time.time()
col_cust.update_one(
    {'_id': 99999},
    {'$pull': {'invoices': {'InvoiceNo': '999999'}}}
)
end_time = time.time()
print(f"Customer-Centric:    Delete (pull) took {end_time - start_time:.6f} seconds.")

# Clean up the test customer
col_cust.delete_one({'_id': 99999})

# Close Connection
client.close()
print("Tests complete. Connection closed.")
