from contextlib import contextmanager

import psycopg2

def log_to_db(action_type, code, error_message):
    db_util = DbUtil()
    db_util.log(action_type, code, error_message)

class DbUtil:
    def __init__(
            self,
            username='dsuleev',
            password='postgres',
            db='mlops_api',
            host='db'
    ):
        self.dbname = db
        self.user = username
        self.password = password
        self.host = host

    def log(self, action_type, code, error_message):
        with self.get_cursor() as c:
            c.execute(
                f'INSERT INTO action_logs (action_type, action_datetime, code, error_message ) '
                f'VALUES ({action_type}, CURRENT_TIMESTAMP, {code}, {error_message})'
            )

    @contextmanager
    def get_cursor(self):
        conn = psycopg2.connect(
            dbname=self.dbname,
            user=self.user,
            password=self.password,
            host=self.host
        )

        cursor = conn.cursor()

        try:
            yield cursor
            conn.commit()
        finally:
            cursor.close()
            conn.close()