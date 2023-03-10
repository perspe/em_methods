from logging.config import fileConfig
import os

# Get module logger
base_path = os.path.dirname(os.path.abspath(__file__))
fileConfig(os.path.join(base_path, "logging.ini"), disable_existing_loggers=False)