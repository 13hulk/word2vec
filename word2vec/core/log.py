from logfire import configure, Logfire

from kit.proxy import ProxyObject, Sentinel

_sentinel_logger = Sentinel()
LOG: Logfire = ProxyObject(_sentinel_logger)  # type: ignore


def init_logger():
    global _sentinel_logger

    _sentinel_logger.obj = _create_logfire()
    print(LOG.config)


def _create_logfire() -> Logfire:
    configure(
        project_name="core",
        service_name="core",
        service_version="0.1.0",
        show_summary=True,
    )
    return Logfire()
