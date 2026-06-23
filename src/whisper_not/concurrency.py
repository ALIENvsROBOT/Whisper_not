import asyncio
from typing import Callable, Optional

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send


class RequestQueueFull(Exception):
    """Raised when an audio request cannot enter the bounded processing queue."""


class AudioRequestLimiter:
    """Bound active requests and waiters without blocking the event loop."""

    def __init__(
        self,
        *,
        max_active: int,
        max_queued: int,
        wait_timeout: float,
    ) -> None:
        if max_active < 1:
            raise ValueError("max_active must be at least 1")
        if max_queued < 0:
            raise ValueError("max_queued must be zero or greater")
        if wait_timeout <= 0:
            raise ValueError("wait_timeout must be greater than zero")

        self._max_active = max_active
        self._max_queued = max_queued
        self._wait_timeout = wait_timeout
        self._active = 0
        self._waiting = 0
        self._condition = asyncio.Condition()

    @property
    def active(self) -> int:
        return self._active

    @property
    def waiting(self) -> int:
        return self._waiting

    async def acquire(self) -> None:
        async with self._condition:
            if self._active < self._max_active:
                self._active += 1
                return
            if self._waiting >= self._max_queued:
                raise RequestQueueFull("The transcription queue is full.")

            self._waiting += 1
            acquired = False
            try:
                await asyncio.wait_for(
                    self._wait_for_available_slot(),
                    timeout=self._wait_timeout,
                )
                self._active += 1
                acquired = True
            except asyncio.TimeoutError as exc:
                raise RequestQueueFull(
                    "Timed out waiting for a transcription slot."
                ) from exc
            finally:
                self._waiting -= 1
                if not acquired and self._active < self._max_active:
                    self._condition.notify(1)

    async def _wait_for_available_slot(self) -> None:
        while self._active >= self._max_active:
            await self._condition.wait()

    async def release(self) -> None:
        async with self._condition:
            if self._active <= 0:
                raise RuntimeError("Cannot release an inactive request limiter.")
            self._active -= 1
            self._condition.notify(1)


class AudioAdmissionMiddleware:
    """Apply admission control before multipart request bodies are consumed."""

    _AUDIO_PATHS = {
        "/v1/audio/transcriptions",
        "/v1/audio/translations",
    }

    def __init__(
        self,
        app: ASGIApp,
        *,
        limiter: AudioRequestLimiter,
        authorize: Optional[Callable[[Optional[str]], Optional[str]]] = None,
    ) -> None:
        self.app = app
        self.limiter = limiter
        self.authorize = authorize

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if (
            scope["type"] != "http"
            or scope.get("method") != "POST"
            or scope.get("path") not in self._AUDIO_PATHS
        ):
            await self.app(scope, receive, send)
            return

        if self.authorize is not None:
            authorization = self._header(scope, b"authorization")
            auth_error = self.authorize(authorization)
            if auth_error is not None:
                await JSONResponse(
                    {"detail": auth_error},
                    status_code=401,
                )(scope, receive, send)
                return

        try:
            await self.limiter.acquire()
        except RequestQueueFull as exc:
            await JSONResponse(
                {"detail": str(exc)},
                status_code=429,
                headers={"Retry-After": "30"},
            )(scope, receive, send)
            return

        released = False

        async def send_with_release(message: Message) -> None:
            nonlocal released
            await send(message)
            if (
                message["type"] == "http.response.body"
                and not message.get("more_body", False)
                and not released
            ):
                released = True
                await self.limiter.release()

        try:
            await self.app(scope, receive, send_with_release)
        finally:
            if not released:
                await self.limiter.release()

    @staticmethod
    def _header(scope: Scope, name: bytes) -> Optional[str]:
        for header_name, value in scope.get("headers", []):
            if header_name.lower() == name:
                return value.decode("latin-1")
        return None
