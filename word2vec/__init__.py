from word2vec.core import LOG


def create_app():
    init_modules()
    LOG.info("Initialized all modules")


def init_modules():
    from word2vec.core import init_logger

    # Instrumentation
    # 1. Logger
    init_logger()
    LOG.info("Initialized logger")
