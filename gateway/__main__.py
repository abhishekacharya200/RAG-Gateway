"""Start the gateway: python -m gateway"""

import ipaddress
import sys

import uvicorn

from .config import load_settings


def _is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def main() -> None:
    settings = load_settings()
    if not _is_loopback(settings.host) and not settings.allow_remote:
        sys.exit(f"Refusing to listen on {settings.host}: other computers could use your Claude subscription "
                 "through this gateway. Use GATEWAY_HOST=127.0.0.1, or set GATEWAY_ALLOW_REMOTE=1 if you "
                 "really mean it.")
    uvicorn.run("gateway.main:app", host=settings.host, port=settings.port, log_level=settings.log_level.lower())


if __name__ == "__main__":
    main()
