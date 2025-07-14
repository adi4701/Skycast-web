import schedule
import time
import os

# Set the time you want the update to run (24-hour format, e.g., '07:00' for 7 AM)
UPDATE_TIME = '07:00'

# Define the job
def job():
    print('Running real-time weather fetch and model retrain...')
    os.system('python fetch_and_train_realtime.py')
    print('Done.')

# Schedule the job
schedule.every().day.at(UPDATE_TIME).do(job)

print(f'Scheduled real-time weather/model update every day at {UPDATE_TIME}. Press Ctrl+C to stop.')

while True:
    schedule.run_pending()
    time.sleep(60) 