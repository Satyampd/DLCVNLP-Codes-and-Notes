import yaml
import time
import os

def read_config(config_path):
    with open(config_path) as config_file:
        content = yaml.safe_load(config_file)
    return content

def get_log_path(log_dir = "logs/fit"):
  uniqueName = time.strftime("log_%Y_%m_%d_%H_%M_%S")
  log_path = os.path.join(log_dir,uniqueName )

  return log_path

def get_unique_file_name(filename):
    unique_filename = time.strftime(f"%Y%m%d_%H%M%S_{filename}")
    return unique_filename
