import logging
import os

logger = logging.getLogger(__name__)


def touch_data_dir():
    path = data_dir_path()
    if not os.path.exists(path):
        logger.info("Creating hiob data directory: '%s'", path)
    os.makedirs(path, exist_ok=True)


def touch_model_dir():
    path = data_file_path('models')
    if not os.path.exists(path):
        logger.info("Creating hiob model directory: '%s'", path)
    os.makedirs(path, exist_ok=True)


def data_dir_path():
    return os.path.abspath(os.path.expanduser(os.path.join("~", ".hiob")))


def data_file_path(path):
    return os.path.join(data_dir_path(), path)


def data_file_exists(path):
    return os.path.exists(data_file_path(path))


def model_file_path(file):
    return os.path.join(data_dir_path(), "models", file)


def open_data_file(file, *args, **kwargs):
    logger.info("opening data file '%s'", file)
    touch_data_dir()
    return open(data_file_path(file), *args, **kwargs)
