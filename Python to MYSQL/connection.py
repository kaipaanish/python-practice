import mysql.connector  

# Create connection
conn = mysql.connector.connect(
    host='localhost',
    user='root',
    password='1001'
)


if conn.is_connected():
    print(" Connected to MySQL Server")

print(conn)
print(conn.is_connected())
