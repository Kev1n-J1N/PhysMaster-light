import os
import sys

from loguru import logger


# Configure logging with loguru
def setup_logging(log_file=None):
    import logging
    import sys
    if log_file is None:
        # Create logs directory if it doesn't exist
        log_dir = os.path.dirname(os.path.abspath(__file__))
        log_file = os.path.join(log_dir, "papermaster.log")

    # Remove default handler
    logger.remove()

    # Add file handler with rotation
    logger.add(
        log_file,
        rotation="10 MB",
        retention=5,
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss:SSS} | {level: <8} | API | {name}:{function}:{line} | {message} | extra={extra}",
        encoding="utf-8",
    )

    # Add console handler with colorful format
    logger.add(
        sys.stdout,
        level="INFO",
        format="<green>{time:YYYY-MM-DD HH:mm:ss:SSS}</green> | <level>{level: <8}</level> | <blue>API</blue> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | <level>{message}</level> | extra={extra}",
        colorize=True,
    )

    # 配置标准库 logging 与 loguru 的集成
    # 这样可以捕获 uvicorn、fastapi 等使用标准库 logging 的日志
    class InterceptHandler(logging.Handler):
        def emit(self, record):
            # 获取对应的 loguru 级别
            try:
                level = logger.level(record.levelname).name
            except ValueError:
                level = record.levelno

            # 找到调用者的帧
            frame, depth = logging.currentframe(), 2
            while frame.f_code.co_filename == logging.__file__:
                frame = frame.f_back
                depth += 1

            logger.opt(depth=depth, exception=record.exc_info).log(
                level, record.getMessage()
            )

    # 拦截标准库的日志记录器
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)

    # 确保 uvicorn 的日志也被拦截
    for name in ["uvicorn", "uvicorn.error", "uvicorn.access", "fastapi"]:
        logging.getLogger(name).handlers = [InterceptHandler()]
        logging.getLogger(name).propagate = False

    # 同时拦截 Celery 日志（当在 API 应用中使用 Celery 时）
    celery_loggers = [
        "celery",
        "celery.app",
        "celery.task",
        "celery.worker",
        "celery.beat",
        "celery.redirected",
        "celery.app.trace",
        "billiard",
        "kombu",
    ]

    for logger_name in celery_loggers:
        celery_logger = logging.getLogger(logger_name)
        if not celery_logger.handlers or not any(
            isinstance(h, InterceptHandler) for h in celery_logger.handlers
        ):
            celery_logger.handlers = [InterceptHandler()]
            celery_logger.propagate = False
            celery_logger.setLevel(logging.INFO)

    logger._configured_for_api = True

    return logger

