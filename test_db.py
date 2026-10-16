from db import get_connection

conn = get_connection()

with conn.cursor() as cur:
    cur.execute(
        """
        SELECT COUNT(*)
        FROM wiki_vector_store
        """
    )

    count = cur.fetchone()[0]

    print("Total records:", count)


conn.close()