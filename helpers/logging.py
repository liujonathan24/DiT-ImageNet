import logging
import os
import json

class JsonFormatter(logging.Formatter):
    def format(self, record):
        json_record = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "message": record.getMessage(),
        }
        return json.dumps(json_record)

def setup_logging(experiment_path):
    log_path = os.path.join(experiment_path, "training.log")
    json_log_path = os.path.join(experiment_path, "training.json.log")

    print(f"Logs will be saved to: {log_path} and {json_log_path}")

    # Set up the logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Create a file handler for traditional logging
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(file_handler)

    # Create a file handler for JSON logging
    json_file_handler = logging.FileHandler(json_log_path)
    json_file_handler.setFormatter(JsonFormatter())
    logger.addHandler(json_file_handler)

    # Create a console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(console_handler)