import logging


_logger = None
_streamHandler = logging.StreamHandler()


def logger():
    return _logger


def start_logging(log_filename=None, logging_level=logging.WARNING):
    global _logger
    _logger = logging.getLogger(__name__)

    current_logging_level = logger().getEffectiveLevel()
    if not current_logging_level or current_logging_level > logging_level:
        logger().setLevel(logging_level)

    if not log_filename:
        handler = _streamHandler
    else:
        logger().removeHandler(_streamHandler)
        handler = logging.FileHandler(str(log_filename))

    handler.setLevel(logging_level)
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - in %(module)s.%(funcName)s (line %(lineno)d): %(message)s')
    handler.setFormatter(formatter)
    logger().addHandler(handler)


start_logging()
