import sqlite3

conn = sqlite3.connect('Scanner.db')

cursor = conn.cursor()

users_table_query = '''
CREATE TABLE Users (
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    Login TEXT NOT NULL,
    Password TEXT NOT NULL,
    TotalScans INTEGER
);
'''
cursor.execute(users_table_query)

scans_table_query = '''
CREATE TABLE Scans (
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    OriginalImage BLOB,
    DenoisedImage BLOB,
    ScannedText TEXT
);
'''
cursor.execute(scans_table_query)

users_scans_table_query = '''
CREATE TABLE UsersScans (
    UserID INTEGER,
    ScanID INTEGER,
    FOREIGN KEY (UserID) REFERENCES Users (ID),
    FOREIGN KEY (ScanID) REFERENCES Scans (ID)
);
'''
cursor.execute(users_scans_table_query)