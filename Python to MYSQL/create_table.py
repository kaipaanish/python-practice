import mysql.connector

# Connect to MySQL and the specific database
conn = mysql.connector.connect(
    host='localhost',
    user='root',
    password='1001',
    database='mydatabase'
)

# ✅ Create the cursor
mycursor = conn.cursor()

# Create table (note: 'class' is a reserved keyword, use 'calss' or backticks if needed)
mycursor.execute('CREATE TABLE IF NOT EXISTS calss (name VARCHAR(50), branch VARCHAR(50), id INT)')

# Show tables
mycursor.execute('SHOW TABLES')

# Print table list
for x in mycursor:
    print(x)
