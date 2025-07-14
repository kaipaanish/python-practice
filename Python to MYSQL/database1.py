import mysql.connector

conn = mysql.connector.connect(
    host='localhost',
    user='root',
    password='1001'
)

mycursor = conn.cursor()

mycursor.execute("SHOW DATABASES")
for x in mycursor:
    print(x)
