import cx_Oracle

class DBManager:
    def __init__(self):
        self.conn = None
        self.get_connection()
    def get_connection(self):
        self.conn = cx_Oracle.connect("ecoala"
                                     ,"ecoala","192.168.0.44:1521/xe")
        return  self.conn
    def __del__(self):
        try:
            print("소멸자")
            if self.conn:
                self.conn.close()
        except Exception as err:
            print("__del__",str(err))
    def insert(self,query,param):
        cursor = self.conn.cursor()
        cursor.execute(query,param)
        self.conn.commit()
        cursor.close()

