import logging
import os
from datetime import datetime, timezone, timedelta

# 한국 시간대
KST = timezone(timedelta(hours=9))

def setup_logger(name="model_logger", log_dir="/app/logs", log_file_prefix="model_log"):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{log_file_prefix}_{datetime.now(KST).strftime('%Y-%m-%d')}.txt")
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    formatter.converter = lambda *args: datetime.now(KST).timetuple()
    
    if not logger.handlers:
        logger.addHandler(file_handler)
    
    return logger
