import sqlite3

conn = sqlite3.connect('data/cost_analytics.db')
cursor = conn.cursor()

# Get all tables
cursor.execute('SELECT name FROM sqlite_master WHERE type="table"')
tables = cursor.fetchall()
print('Tables:', tables)

# Check row count for each table with detailed info
for table in tables:
    table_name = table[0]
    cursor.execute(f'SELECT COUNT(*) FROM {table_name}')
    count = cursor.fetchone()
    print(f'{table_name}: {count[0]} rows')

    # Show schema
    cursor.execute(f'PRAGMA table_info({table_name})')
    columns = cursor.fetchall()
    print(f'  Columns: {[col[1] for col in columns]}')

    # Show sample data if any
    if count[0] > 0:
        cursor.execute(f'SELECT * FROM {table_name} LIMIT 3')
        sample_rows = cursor.fetchall()
        print(f'  Sample rows: {sample_rows}')

conn.close()
