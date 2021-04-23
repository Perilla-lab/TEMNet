import sqlite3
import os

# ********* DATABASE HELPER FUNCTIONS *********
""" 
db_connect: Connect to SQL Database file, prints the status of the database connection
Inputs: db, the name of the .db file the user wishes to connect to
Outputs: None
"""
def db_connect(db):
  conn = None
  try:
    conn = sqlite3.connect(db)
    print("database connection successful")
  except sqlite3.Error:
    print("Error: database connection unsuccessful")
  return conn

""" 
db_query: Interacts with SQL Database file in order to return the filenames of the relevant particle images based on user input 
Inputs: query, the desired SQL query, and conn, the connection  to the database
Outputs: ans, an array of queried information from the database
"""
def db_query(query, conn):
  crsr = conn.cursor()
  crsr.execute(query)
  ans = crsr.fetchall()
  print("Returned from query", query)
  for i in ans: 
    print(i)
  return ans

""" 
get_max_dims: Returns the maximum dimensions of all files in the database 
Inputs: conn, the database connection
Outputs: width and length, the maximum measurements from all files in the database
"""
def get_max_dims(conn):
  #os.chdir('/home/hbeatson/jupyter_runtime_dir/HIV/')
  crsr = conn.cursor()
  crsr.execute('SELECT MAX(pxWidth) FROM imgs')
  width = crsr.fetchone()
  width = width[0]
  crsr.execute('SELECT MAX(pxLen) FROM imgs')
  length = crsr.fetchone()
  length = length[0]
  print("Maximum Image Resolution (px): ", width, "x", length)
  return width, length