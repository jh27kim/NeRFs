import logging
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig


def create_logger(logger_name, cfg):
    # Create Logger
    logger = logging.getLogger(logger_name)
    
    # Check handler exists
    if len(logger.handlers) > 0:
        return logger # Logger already exists
    
    if cfg.log == "debug":
        log_level = logging.DEBUG
    elif cfg.log == "info":
        log_level = logging.INFO
    elif cfg.log == "WARN":
        log_level = logging.WARN
    elif cfg.log == "ERROR":
        log_level = logging.ERROR
 
    logger.setLevel(log_level)
 
    formatter = logging.Formatter('\n[%(levelname)s|%(name)s|%(filename)s:%(lineno)s] %(asctime)s > %(message)s')
    
    # Create Handlers
    streamHandler = logging.StreamHandler()
    streamHandler.setLevel(log_level)
    streamHandler.setFormatter(formatter)
 
    # logger.addHandler(streamHandler)

    return logger