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

sql = "INSERT INTO calss (name, branch, id) VALUES (%s, %s, %s)"

val = [
    ('John', 'Computer Science', 1),
    ('Jane', 'Electrical Engineering', 2),
    ('Mike', 'Mechanical Engineering', 3)]

mycursor.executemany(sql, val)
conn.commit()
print(mycursor.rowcount, "was inserted.")