import sqlite3
def get_candidate(candidate_id):
    conn = sqlite3.connect("hr_ai.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM candidates WHERE candidate_id = ?", (candidate_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        columns = [col[0] for col in cursor.description]
        return dict(zip(columns, row))
    return None


if __name__ == "__main__":
    candidate = get_candidate("CAND002")
    print(candidate)
