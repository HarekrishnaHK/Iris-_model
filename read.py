import sqlite3
import time

def read_new_predictions():
    # Connect to the SQLite database
    conn = sqlite3.connect('db.sqlite')
    cursor = conn.cursor()
    last_seen_id = 0  # Keeps track of the last read row

    print("🔁 Monitoring new predictions every 90 seconds...\n")

    while True:
        # Get all new predictions added since the last check
        cursor.execute("SELECT * FROM predictions WHERE id > ?", (last_seen_id,))
        new_predictions = cursor.fetchall()

        if new_predictions:
            print(f"\n🔔 New predictions at {time.strftime('%H:%M:%S')}:\n")
            for row in new_predictions:
                id, sl, sw, pl, pw, label = row
                print(f"🧾 ID {id}: [Sepal={sl}, {sw} | Petal={pl}, {pw}] → 🌸 {label}")
                last_seen_id = max(last_seen_id, id)
        else:
            print(f"⏰ No new predictions at {time.strftime('%H:%M:%S')}")

        # Wait 90 seconds before checking again
        time.sleep(90)

if __name__ == "__main__":
    read_new_predictions()
