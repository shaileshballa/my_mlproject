import logging
import os
from datetime import datetime

# Generate log file name with timestamp
LOG_FILE = f"{datetime.now().strftime('%m-%d_%Y__%H_%M_%S')}.log"

# Define log folder and create it
log_path = os.path.join(os.getcwd(), 'logs')
os.makedirs(log_path, exist_ok=True)

# Create full log file path
LOG_FILE_PATH = os.path.join(log_path, LOG_FILE)

# Configure logging
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format='[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    filemode='w'
)
