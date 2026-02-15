import sqlite3, json
from pathlib import Path
from sys import exit

db = Path('data/orchestrator.db')
if not db.exists():
    print(json.dumps({"error": f"DB not found: {db}"}))
    exit(1)

con = sqlite3.connect(str(db))
cur = con.cursor()

res = {"db": str(db), "exists": True}
try:
    cur.execute('SELECT COUNT(*) FROM calibration_log')
    res["total"] = cur.fetchone()[0]
    cur.execute('''
        SELECT calibration_id,timestamp,operator_id,calibrant_material,overall_pass
        FROM calibration_log
        WHERE timestamp >= datetime('now','-24 hours')
        ORDER BY timestamp DESC LIMIT 10
    ''')
    rows = cur.fetchall()
    res["recent"] = [
        {
            "calibration_id": r[0],
            "timestamp": r[1],
            "operator": r[2],
            "material": r[3],
            "overall_pass": bool(r[4])
        } for r in rows
    ]
finally:
    con.close()

print(json.dumps(res, indent=2))
