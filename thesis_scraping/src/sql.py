import psycopg2

class SQLConnection:
    def __init__(self, host, port, database, user, password):
        self.conn = psycopg2.connect(
            host=host,
            port=port,
            database=database,
            user=user,
            password=password
        )
        self.cursor = self.conn.cursor()

    def create_table(self):
        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS firms (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            website TEXT,
            thesis TEXT,
            country TEXT,
            founded TEXT,
            industry TEXT,
            linkedin_url TEXT,
            locality TEXT,
            region TEXT,
            size TEXT
        );
        """)
        self.conn.commit()

    def save_firm_to_db(self, id: str, name: str, website: str, thesis: str,
                        country: str, founded: str, industry: str,
                        linkedin_url: str, locality: str, region: str, size: str):
        try:
            self.cursor.execute("""
                INSERT INTO firms (id, name, website, thesis, country, founded, industry, linkedin_url, locality, region, size)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO NOTHING;
            """, (id, name, website, thesis, country, founded, industry, linkedin_url, locality, region, size))
            self.conn.commit()
            print(f"Saved: {name}")
        except Exception as e:
            print("Error saving firm:", e)

    def select_all(self):
        try:
            self.cursor.execute("SELECT * FROM firms;")
            rows = self.cursor.fetchall()
            for row in rows:
                print(row)
        except Exception as e:
            print("Error selecting rows:", e)


    def drop_table(self, table_name="firms"):
        try:
            self.cursor.execute(f"DROP TABLE IF EXISTS {table_name};")
            self.conn.commit()
            print(f"Table '{table_name}' dropped.")
        except Exception as e:
            print(f"Error dropping table '{table_name}':", e)


    def close(self):
        self.cursor.close()
        self.conn.close()