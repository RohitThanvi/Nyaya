"""Audit log middleware — records all API calls for compliance."""
import json
import logging
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("audit")


class AuditLogMiddleware(BaseHTTPMiddleware):
    SKIP_PATHS = {"/api/v1/health", "/api/docs", "/api/redoc", "/openapi.json"}

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        path = request.url.path
        if path not in self.SKIP_PATHS and not path.startswith("/api/docs"):
            logger.info(
                json.dumps({
                    "method": request.method,
                    "path": path,
                    "status": response.status_code,
                    "ip": request.client.host if request.client else "unknown",
                    "user_agent": request.headers.get("user-agent", ""),
                })
            )
        return response
