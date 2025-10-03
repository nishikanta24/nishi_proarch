import sqlite3

conn = sqlite3.connect('data/cost_analytics.db')
cursor = conn.cursor()

# Get all tables
cursor.execute('SELECT name FROM sqlite_master WHERE type="table"')
tables = cursor.fetchall()
print('Tables:', tables)

# Check row count for each table
for table in tables:
    cursor.execute(f'SELECT COUNT(*) FROM {table[0]}')
    count = cursor.fetchone()
    print(f'{table[0]}: {count[0]} rows')

conn.close()
