'''Logging Setup

This module defines a function called setup_logging which is currently
called in fetch_and_alert.py which allows debug messages to be sent to
a logging file and for warning and critical messages to be sent to
the console.
'''
import logging


def setup_logging(log_file):
    '''Enables logging messages to print to a file and to console

    This function configures all debug messages to print to
    a .log file and all warning and critical messages to print
    to the console. It also defines the format that these messages
    will be in which includes their time. It's currently configured
    to append the same file with each new run so all info from each
    separate run will be in the same place.

    Args:

    log_file: the name of the file that will have the logging messages
    appended to it with each run
    '''
    logging.basicConfig(filename=log_file,
                        filemode='a',
                        level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)

    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
