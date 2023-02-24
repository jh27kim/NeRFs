import hydra
from logs.create_logger import create_logger
from domain.Nerf import NeRF
import os

@hydra.main(
    version_base=None,
    config_path="static",  # config file search path is relative to this script
    config_name="config",
)
def main(cfg):
    cmd = 'echo start running > ./logs/main.log'
    os.system(cmd)

    logger = create_logger("NeRF-logger", cfg)
    logger.info(f"Running model : {cfg.network.model}")

    if cfg.network.model == "nerf":
        m = NeRF(cfg, logger)
    else:
        raise Exception("Model not available : ", cfg.network.model)
    
    m.run()

if __name__ == "__main__":
    main()
