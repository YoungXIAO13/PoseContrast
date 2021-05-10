import os
import logging
import time


class TerminalColors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def get_time():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def print_info(s):
    print(TerminalColors.OKBLUE + "[" + get_time() + "] " + str(s) + TerminalColors.ENDC)


def print_warning(s):
    print(TerminalColors.WARNING + "[" + get_time() + "] WARN " + str(s) + TerminalColors.ENDC)


def print_error(s):
    print(TerminalColors.FAIL + "[" + get_time() + "] ERROR " + str(s) + TerminalColors.ENDC)


def get_logger(log_dir, name):
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    file_path = os.path.join(log_dir, "{}.log".format(name))
    hdlr = logging.FileHandler(file_path)
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s")
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)
    return logger
