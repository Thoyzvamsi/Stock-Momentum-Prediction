from collections.abc import Iterable
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import json
import time
from threading import Lock

from groq import APIConnectionError, APIStatusError, APITimeoutError


FALLBACK_MODELS = (
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
)
MIN_GROQ_REQUEST_INTERVAL_SECONDS = 30
DEFAULT_RATE_LIMIT_COOLDOWN_SECONDS = 60


class GroqInvalidResponseError(ValueError):
    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


class GroqCooldownError(RuntimeError):
    def __init__(self, retry_after_seconds: float):
        self.retry_after_seconds = max(1, int(retry_after_seconds + 0.999))
        super().__init__(
            f"Groq request cooldown is active for {self.retry_after_seconds} seconds."
        )


class GroqRequestLimiter:
    def __init__(
        self,
        min_interval_seconds: float = MIN_GROQ_REQUEST_INTERVAL_SECONDS,
        cooldown_seconds: float = DEFAULT_RATE_LIMIT_COOLDOWN_SECONDS,
        clock=time.monotonic,
    ):
        self.min_interval_seconds = min_interval_seconds
        self.cooldown_seconds = cooldown_seconds
        self._clock = clock
        self._next_allowed_at = 0.0
        self._blocked_until = 0.0
        self._lock = Lock()

    def acquire(self) -> None:
        with self._lock:
            now = self._clock()
            wait_seconds = max(self._next_allowed_at, self._blocked_until) - now
            if wait_seconds > 0:
                raise GroqCooldownError(wait_seconds)
            self._next_allowed_at = now + self.min_interval_seconds

    def defer_after_rate_limit(
        self, retry_after_seconds: float | None = None
    ) -> float:
        cooldown = max(
            self.cooldown_seconds,
            retry_after_seconds or self.cooldown_seconds,
        )
        with self._lock:
            now = self._clock()
            self._blocked_until = max(self._blocked_until, now + cooldown)
            return self._blocked_until - now


def rate_limit_retry_after(error: Exception) -> float | None:
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", {})
    value = (
        headers.get("retry-after") or headers.get("Retry-After")
        if headers
        else None
    )
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        try:
            retry_at = parsedate_to_datetime(value)
            if retry_at.tzinfo is None:
                retry_at = retry_at.replace(tzinfo=timezone.utc)
            return max(
                0.0,
                (retry_at - datetime.now(timezone.utc)).total_seconds(),
            )
        except (TypeError, ValueError, OverflowError):
            return None


def select_available_model(
    available_models: Iterable[str], preferred_model: str
) -> str | None:
    available = set(available_models)
    candidates = dict.fromkeys((preferred_model, *FALLBACK_MODELS))
    return next((model for model in candidates if model in available), None)


def parse_groq_sentiment_response(
    content: str | None, finish_reason: str | None = None
) -> dict:
    if finish_reason == "length":
        raise GroqInvalidResponseError(
            "The response was truncated. Increase the response token limit."
        )
    if not content or not content.strip():
        raise GroqInvalidResponseError("The model returned an empty response.")

    response = content.strip()
    if response.startswith("```"):
        response = response.removeprefix("```json").removeprefix("```").strip()
        response = response.removesuffix("```").strip()

    start = response.find("{")
    if start == -1:
        raise GroqInvalidResponseError(
            "The model response did not contain a JSON object."
        )

    try:
        result, _ = json.JSONDecoder().raw_decode(response[start:])
    except json.JSONDecodeError as error:
        raise GroqInvalidResponseError(
            "The model returned malformed or truncated JSON."
        ) from error

    if not isinstance(result, dict):
        raise GroqInvalidResponseError("The model response was not a JSON object.")

    required_fields = {
        "overall_sentiment": str,
        "sentiment_score": int,
        "key_themes": list,
        "bullish_factors": list,
        "bearish_factors": list,
        "short_term_outlook": str,
        "risk_factors": list,
        "confidence": str,
    }
    for field, expected_type in required_fields.items():
        value = result.get(field)
        if not isinstance(value, expected_type) or (
            expected_type is int and isinstance(value, bool)
        ):
            raise GroqInvalidResponseError(
                f"The model response has a missing or invalid '{field}' field."
            )

    if result["overall_sentiment"] not in {"BULLISH", "BEARISH", "NEUTRAL", "MIXED"}:
        raise GroqInvalidResponseError(
            "The model returned an unsupported sentiment value."
        )
    if not -100 <= result["sentiment_score"] <= 100:
        raise GroqInvalidResponseError(
            "The model returned a sentiment score outside -100 to +100."
        )
    if result["confidence"] not in {"HIGH", "MEDIUM", "LOW"}:
        raise GroqInvalidResponseError(
            "The model returned an unsupported confidence value."
        )

    return result


def groq_error_message(error: Exception) -> str:
    if isinstance(error, GroqCooldownError):
        return (
            f"Groq is cooling down. Try again in "
            f"{error.retry_after_seconds} seconds."
        )
    if isinstance(error, APITimeoutError):
        return "Groq timed out. Check your connection and try again."
    if isinstance(error, APIConnectionError):
        return (
            "Could not connect to Groq. Check your internet connection, "
            "VPN, or firewall, then try again."
        )
    status_code = getattr(error, "status_code", None)
    if status_code in (401, 403):
        return "Groq authentication failed. Check or replace GROQ_API_KEY."
    if status_code == 429:
        return "Groq rate limit reached. Wait briefly before trying again."
    if status_code == 404:
        return (
            "Groq could not find the requested model or endpoint. "
            "Check GROQ_MODEL and the models available to your key."
        )
    if status_code == 400:
        return (
            "Groq rejected the request. Check the model and request parameters."
        )
    if status_code is not None and status_code >= 500:
        return "Groq is temporarily unavailable. Try again later."
    if status_code is not None:
        return f"Groq request failed (HTTP {status_code})."
    if isinstance(error, ValueError):
        if isinstance(error, GroqInvalidResponseError):
            return f"Groq returned an unusable response: {error.reason}"
        return "Groq returned an invalid response. Check the app logs."
    if isinstance(error, APIStatusError):
        return "Groq returned an API error. Check the app logs for details."
    return f"Groq request failed ({type(error).__name__}). Check the app logs."
