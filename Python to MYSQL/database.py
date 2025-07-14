import mysql.connector

conn = mysql.connector.connect(
    host='localhost',
    user='root',
    password='1001'
)

if conn.is_connected():
    print("Connected to MySQL Server")

print(conn)
print(conn.is_connected())

mycursor = conn.cursor()

# Correct SQL syntax
mycursor.execute("CREATE DATABASE mydatabase")

print("Database created successfully.")
