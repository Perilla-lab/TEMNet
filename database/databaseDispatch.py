# HIV Database Data Dispatch Python Script
# Perilla Labs
# Hagan Beatson Jan 2020

# Module used for a lightweight, efficient version of an SQL database
import sqlite3
import os
import skimage
from skimage import io

# Establish connection to local .db database file
connection = sqlite3.connect("hivimgs.db")
crsr = connection.cursor()

# ******** Helper Functions ********
# **********************************

# get_cellnum: Get cell number based on file name
# Inputs: file, the path of the file whose name it is reading
# Outputs: post, a string corresponding to the cell number of the file
# Works correctly as of 1/14/2020


def get_cellnum(file):
    pre = file.lstrip('emi')
    post = pre.split("_")[0]
    return post

# get_classification: Get image classification based on file name and returns it as a string (Eccentric, Immature, Mature)
# Inputs: file, the path of the file whose name it is reading
# Outputs: char, the leading char in the filename corresponding to its classification
# Works correctly as of 1/14/2020


def get_classification(file):
    char = ''
    if file.startswith('e'):
        char = 'e'
    elif file.startswith('i'):
        char = 'i'
    elif file.startswith('m'):
        char = 'm'
    return char

# get_EID: Get experiment ID based on file name
# Inputs: file, the path of the file whose name it is reading
# Outputs: post, a string corresponding to the experiment ID of the file
# Works correctly as of 1/14/2020


def get_EID(file):
    pre = file.split("_")[1]
    post = pre.rstrip(".tif")
    return post

# get_dims: Get length and width from image in pixels
# Inputs: file, the path of the file whose dimensions it is reading
# Outputs: width and height, the corresponding dimensions of the file in pixels
#


def get_dims(file):
    img = io.imread(file)
    width, height = img.shape
    return width, height

# get_dims_nm: Returns the length and width from image in nanometers
# Inputs: width and height, two ints that are converted
# Outputs: widthNM and heightNM, the respective measurements in nanometers instead of pixels
# Works correctly as of 1/14/2020


def get_dims_nm(width, height):
    widthNM = round(width * 0.561, 4)
    heightNM = round(height * 0.561, 4)
    return widthNM, heightNM


def create_dispatch(conn, dispatch):
    sql = '''INSERT INTO imgs VALUES (?,?,?,?,?,?,?,?) '''
    cur = conn.cursor()
    cur.execute(sql, dispatch)
    return cur.lastrowid

# fill_table: Fill table with relevant data based on the directory you choose
# Inputs: path, the path for the directory whose contents are being committed to the database
# Testing helper functions before commiting them to database


def fill_table(path):
    for im in os.listdir(path):
        if(".tif" in im):
            classification = get_classification(im)
            EID = get_EID(im)
            num = get_cellnum(im)
            width, height = get_dims(im)
            widthNM, heightNM = get_dims_nm(width, height)
            dispatch = (im, num, classification, EID,
                        height, width, heightNM, widthNM)
            create_dispatch(connection, dispatch)
	    print("Dispatched", im)
        else:
            print("Incorrect file type!")

# ******** Main Function ********
# *******************************


# path = 'C:\\Users\\hbeatson\\Documents\\Datasets\\sample_imgs'  #Can be changed to whatever directory you wish to import into a DB
path = '/home/hbeatson/jupyter_runtime_dir/HIV/sample_imgs'
os.chdir(path)
fill_table(path)

connection.commit()
connection.close()
