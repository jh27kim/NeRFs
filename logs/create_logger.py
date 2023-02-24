import logging
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig


def create_logger(logger_name, cfg):
    # Create Logger
    logger = logging.getLogger(logger_name)
    logger.propagate = False
    
    # Check handler exists
    if len(logger.handlers) > 0:
        return logger # Logger already exists
    
    if cfg.logs.level == "debug":
        log_level = logging.DEBUG
    elif cfg.logs.level == "info":
        log_level = logging.INFO
    elif cfg.logs.level == "WARN":
        log_level = logging.WARN
    elif cfg.logs.level == "ERROR":
        log_level = logging.ERROR
    
    formatter = logging.Formatter('\n[%(levelname)s|%(name)s|%(filename)s:%(lineno)s] %(asctime)s > %(message)s')
    
    # Create Handlers
    streamHandler = logging.StreamHandler()
    streamHandler.setLevel(log_level)
    streamHandler.setFormatter(formatter)

    fileHandler = logging.FileHandler("./logs/main.log")
    fileHandler.setFormatter(formatter)
 
    logger.addHandler(streamHandler)
    logger.addHandler(fileHandler)

    logger.setLevel(log_level)

    return logger