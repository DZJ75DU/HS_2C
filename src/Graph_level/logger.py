import logging
import os


class Logger:
    def __init__(self, log_file):
        self.log_file = log_file
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        formatter_with_time = logging.Formatter('%(asctime)s \n %(message)s')
        formatter_without_time = logging.Formatter('%(message)s')
        self.file_handler = logging.FileHandler(log_file)
        self.file_handler.setLevel(logging.INFO)
        self.console_handler = logging.StreamHandler()
        self.console_handler.setLevel(logging.INFO)
        self.logger.addHandler(self.file_handler)
        self.logger.addHandler(self.console_handler)
        self.formatter_with_time = formatter_with_time
        self.formatter_without_time = formatter_without_time

    def log(self, message, include_time=True):
        if include_time:
            self.file_handler.setFormatter(self.formatter_with_time)
            self.console_handler.setFormatter(self.formatter_with_time)
        else:
            self.file_handler.setFormatter(self.formatter_without_time)
            self.console_handler.setFormatter(self.formatter_without_time)

        self.logger.info(message)
