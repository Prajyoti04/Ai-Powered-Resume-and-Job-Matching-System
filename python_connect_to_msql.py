import mysql.connector
import streamlit as st

def get_connection():
    try:
        conn = mysql.connector.connect(
            host=st.secrets["MYSQLHOST"],
            user=st.secrets["MYSQLUSER"],
            password=st.secrets["MYSQLPASSWORD"],
            database=st.secrets["MYSQLDATABASE"],
            port=st.secrets["MYSQLPORT"]
        )
        print("✅ Connection successfully created!")
        return conn
    except mysql.connector.Error as err:
        print(f"❌ Error: {err}")
        return None

# Optional test run
if __name__ == "__main__":
    conn = get_connection()
    if conn:
        conn.close()


