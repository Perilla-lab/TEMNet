# HIV Database Creator Python Script
# Perilla Labs
# Hagan Beatson Jan 2020

#Module used for a lightweight, efficient version of an SQL database
import sqlite3

#Establish connection to local .db database file
connection = sqlite3.connect("hivimgs.db")
crsr = connection.cursor()

#Initialize columns in table
create_table_cmd = """CREATE TABLE imgs (
    filename VARCHAR(10) PRIMARY KEY,
    cellnum VARCHAR(3),      
    class VARCHAR(10),
    EID VARCHAR(10),
    pxLen INTEGER,
    pxWidth INTEGER,
    nmLen FLOAT,
    nmWidth FLOAT);"""
crsr.execute(create_table_cmd)
connection.commit()
connection.close()
