import yaml
import logging.config
from conf import settings

def init_logs():
    with open('logger/config.yml') as cfg:
        config = yaml.safe_load(cfg.read())

    config['root']['level'] = settings.ROOT_LOG_LEVEL
    logging.config.dictConfig(config)
