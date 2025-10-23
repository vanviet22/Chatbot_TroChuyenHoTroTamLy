import os
import mysql.connector
from dotenv import load_dotenv

load_dotenv()

MYSQL_HOST = os.getenv("MYSQL_HOST")
MYSQL_USER = os.getenv("MYSQL_USER")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")
MYSQL_DB_NAME = os.getenv("MYSQL_DB_NAME")


class MySQLConnection:
    """Lớp tiện ích để kết nối MySQL và tái sử dụng."""

    def __init__(self):
        try:
            self.conn = mysql.connector.connect(
                host=MYSQL_HOST,
                user=MYSQL_USER,
                password=MYSQL_PASSWORD,
                database=MYSQL_DB_NAME
            )
            self.cursor = self.conn.cursor(dictionary=True)
            print(f"✅ Kết nối MySQL thành công tới database: {MYSQL_DB_NAME}")
        except mysql.connector.Error as err:
            print(f"❌ Lỗi kết nối MySQL: {err}")
            self.conn = None
            self.cursor = None

    def get_connection(self):
        """Trả về (conn, cursor)."""
        return self.conn, self.cursor

    def get_table(self, table_name: str):
        """Trả về tuple (conn, cursor, table_name) để module khác dùng tiện hơn."""
        if not self.conn or not self.cursor:
            raise ConnectionError("⚠️ Kết nối MySQL chưa được khởi tạo hoặc đã bị đóng.")
        return self.conn, self.cursor, table_name

    def close(self):
        """Đóng kết nối MySQL."""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
        print("🔒 Đã đóng kết nối MySQL.")


# Instance kết nối toàn cục
mysql = MySQLConnection()

def get_table(table_name: str):
    """Hàm tiện ích chung cho mọi bảng"""
    return mysql.get_table(table_name)

# bảng cache
def get_cache_table():
    conn, cursor, _ = get_table("cache")
    return conn, cursor
