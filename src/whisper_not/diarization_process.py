import logging
import multiprocessing
import os
import threading
import traceback
from multiprocessing.connection import Connection
from typing import Any, Optional


class DiarizationProcessError(RuntimeError):
    """Raised when the isolated diarization worker fails."""


def _worker_main(connection: Connection) -> None:
    log_level = getattr(
        logging,
        os.environ.get("WHISPER_LOG_LEVEL", "INFO").upper(),
        logging.INFO,
    )
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    from whisper_not import diarization

    try:
        while True:
            request = connection.recv()
            operation = request["operation"]
            if operation == "shutdown":
                return

            try:
                if operation == "load":
                    pipeline = diarization.load(**request["kwargs"])
                    result: Any = {"sample_rate": pipeline.sample_rate}
                elif operation == "diarize":
                    result = diarization.diarize(
                        request["audio_path"],
                        **request["kwargs"],
                    )
                elif operation == "ping":
                    result = "pong"
                else:
                    raise ValueError(f"Unsupported diarization operation: {operation}")
                connection.send({"ok": True, "result": result})
            except Exception as exc:  # noqa: BLE001
                connection.send(
                    {
                        "ok": False,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
    except EOFError:
        return
    finally:
        connection.close()


class DiarizationProcess:
    """Own a lazy, persistent diarization subprocess isolated from CTranslate2."""

    def __init__(self, *, request_timeout: float = 7200.0) -> None:
        if request_timeout <= 0:
            raise ValueError("request_timeout must be greater than zero")
        self._request_timeout = request_timeout
        self._context = multiprocessing.get_context("spawn")
        self._connection: Optional[Connection] = None
        self._process = None
        self._lock = threading.Lock()

    def load(self, **kwargs) -> dict:
        return self._request("load", kwargs=kwargs)

    def ping(self) -> str:
        return self._request("ping")

    def diarize(self, audio_path: str, **kwargs):
        return self._request(
            "diarize",
            audio_path=audio_path,
            kwargs=kwargs,
        )

    def close(self) -> None:
        with self._lock:
            if self._process is None:
                return
            if self._process.is_alive() and self._connection is not None:
                try:
                    self._connection.send({"operation": "shutdown"})
                    self._process.join(timeout=5)
                except (BrokenPipeError, EOFError, OSError):
                    pass
            self._reset_worker(force=True)

    def _request(self, operation: str, **payload):
        with self._lock:
            self._ensure_worker()
            request = {"operation": operation, **payload}
            try:
                self._connection.send(request)
                if not self._connection.poll(self._request_timeout):
                    raise DiarizationProcessError(
                        f"Diarization worker timed out after "
                        f"{self._request_timeout:g} seconds."
                    )
                response = self._connection.recv()
            except (BrokenPipeError, EOFError, OSError) as exc:
                exit_code = (
                    self._process.exitcode
                    if self._process is not None
                    else "unknown"
                )
                self._reset_worker(force=True)
                raise DiarizationProcessError(
                    f"Diarization worker exited unexpectedly "
                    f"(exit code {exit_code})."
                ) from exc
            except DiarizationProcessError:
                self._reset_worker(force=True)
                raise

            if not response.get("ok"):
                raise DiarizationProcessError(
                    f"{response.get('error', 'Unknown diarization error')}\n"
                    f"{response.get('traceback', '')}".rstrip()
                )
            return response["result"]

    def _ensure_worker(self) -> None:
        if self._process is not None and self._process.is_alive():
            return
        self._reset_worker(force=False)
        parent_connection, child_connection = self._context.Pipe()
        process = self._context.Process(
            target=_worker_main,
            args=(child_connection,),
            name="whisper-diarization",
            daemon=True,
        )
        process.start()
        child_connection.close()
        self._connection = parent_connection
        self._process = process

    def _reset_worker(self, *, force: bool) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None
        if self._process is not None:
            if force and self._process.is_alive():
                self._process.terminate()
            self._process.join(timeout=5)
            if self._process.is_alive():
                self._process.kill()
                self._process.join(timeout=5)
            self._process.close()
            self._process = None
