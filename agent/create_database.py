import sqlite3

# Create/connect to local database
conn = sqlite3.connect("hr_ai.db")
cursor = conn.cursor()

# Create single table for candidates
cursor.execute("""
CREATE TABLE IF NOT EXISTS candidates (
    candidate_id TEXT PRIMARY KEY,
    name TEXT,
    current_role TEXT,
    years_experience REAL,
    skills TEXT,
    degree TEXT,
    location TEXT,
    referral_flag INTEGER,
    english_level TEXT,
    skills_coverage REAL,
    experience_band TEXT,
    location_match TEXT
)
""")

# Example: insert a candidate
cursor.execute("""
INSERT OR REPLACE INTO candidates 
(candidate_id, name, current_role, years_experience, skills, degree, location, referral_flag, english_level, skills_coverage, experience_band, location_match)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
""", ("CAND002", "yassine lajnaoudi", "Data Scientist", 6, "Docker, Kubernetes, AWS, Linux", "Master", "Casablanca", 1, "B2", 90, "5-10", "Local"))

conn.commit()
conn.close()
