import mysql.connector

conn = mysql.connector.connect(host='localhost', username='newuser', password='newpassword', database='codewthme')

my_cursor = conn.cursor()

conn.commit()
conn.close()

print("connection successfully created!")
