import hydra
import logging

logger = logging.getLogger()


if __name__ == "__main__":
    root_logger = logging.getLogger()
    root_logger.debug("디버그")
    root_logger.info("정보")
    root_logger.error("오류")