import logging
import logging.config
from src.config.settings import get_settings

def setup_logging():
    """Set up logging configuration for the user management service."""
    settings = get_settings()
    
    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": "[{asctime}] {levelname:8} {name}: {message}",
                "style": "{",
                "datefmt": "%H:%M:%S",
            },
            "json": {
                "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
                "format": "%(asctime)s %(name)s %(levelname)s %(message)s %(request_id)s",
            },
        },
        "handlers": {
            "default": {
                "formatter": "default",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
            },
        },
        "root": {
            "level": settings.log_level,
            "handlers": ["default"],
        },
        "loggers": {
            "uvicorn": {
                "level": "INFO",
                "handlers": ["default"],
                "propagate": False,
            },
            "uvicorn.error": {
                "level": "INFO",
                "handlers": ["default"],
                "propagate": False,
            },
            "uvicorn.access": {
                "level": "INFO",
                "handlers": ["default"],
                "propagate": False,
            },
        },
    }
    
    logging.config.dictConfig(logging_config)