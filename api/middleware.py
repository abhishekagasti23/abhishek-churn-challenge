"""
Simple request logging middleware.
"""

import logging
import time

from starlette.middleware.base import (
    BaseHTTPMiddleware,
)

from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger(
    "api.middleware"
)


class RequestLoggingMiddleware(
    BaseHTTPMiddleware
):
    """
    Log request method,
    status code,
    and response time.
    """

    async def dispatch(
        self,
        request: Request,
        call_next,
    ) -> Response:

        start_time = time.perf_counter()

        # Process request
        response = await call_next(
            request
        )

        elapsed_ms = (
            time.perf_counter()
            - start_time
        ) * 1000

        logger.info(
            "%s %s -> %d (%.1f ms)",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )

        # Add response timing header
        response.headers[
            "X-Process-Time-Ms"
        ] = f"{elapsed_ms:.2f}"

        return response