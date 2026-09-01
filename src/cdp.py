import os
# Gets the exact directory path where cdp.py lives on the server
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Attaches the file name to that directory path
DATA_PATH = os.path.join(BASE_DIR, 'cardio_train.csv')

# Your existing pandas code
df = pd.read_csv(DATA_PATH, sep=';')
