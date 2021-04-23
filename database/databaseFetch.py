#Database Fetcher
#Hagan Beatson
import sqlite3

connection = sqlite3.connect("hivimgs.db")

crsr = connection.cursor()

crsr.execute("SELECT * FROM imgs")
ans = crsr.fetchall()

for i in ans:
    print(i)
