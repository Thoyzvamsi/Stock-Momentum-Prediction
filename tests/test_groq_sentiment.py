from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
import unittest

import httpx
from groq import APIConnectionError, APITimeoutError
from unittest.mock import patch

from groq_sentiment import (
    GroqCooldownError,
    GroqInvalidResponseError,
    GroqRequestLimiter,
    groq_error_message,
    parse_groq_sentiment_response,
    rate_limit_retry_after,
    select_available_model,
)


class GroqSentimentTests(unittest.TestCase):
    class AuthenticationError(Exception):
        status_code = 401

    def test_prefers_configured_model_when_available(self):
        model = select_available_model(
            ["llama-3.1-8b-instant", "custom-model"], "custom-model"
        )
        self.assertEqual(model, "custom-model")

    def test_uses_first_available_fallback(self):
        model = select_available_model(
            ["openai/gpt-oss-20b", "llama-3.1-8b-instant"], "retired-model"
        )
        self.assertEqual(model, "llama-3.1-8b-instant")

    def test_returns_none_when_no_supported_model_is_available(self):
        self.assertIsNone(select_available_model(["unrelated-model"], "retired-model"))

    def test_authentication_errors_are_actionable(self):
        self.assertEqual(
            groq_error_message(self.AuthenticationError("Unauthorized")),
            "Groq authentication failed. Check or replace GROQ_API_KEY.",
        )

    def test_request_limiter_enforces_minimum_interval(self):
        now = [100.0]
        limiter = GroqRequestLimiter(min_interval_seconds=30, clock=lambda: now[0])
        limiter.acquire()

        with self.assertRaises(GroqCooldownError) as raised:
            limiter.acquire()
        self.assertEqual(raised.exception.retry_after_seconds, 30)

        now[0] += 30
        limiter.acquire()

    def test_rate_limit_sets_cooldown_at_least_retry_after(self):
        now = [100.0]
        limiter = GroqRequestLimiter(
            min_interval_seconds=30, cooldown_seconds=60, clock=lambda: now[0]
        )
        limiter.acquire()
        limiter.defer_after_rate_limit(120)
        now[0] += 60

        with self.assertRaises(GroqCooldownError) as raised:
            limiter.acquire()
        self.assertEqual(raised.exception.retry_after_seconds, 60)
        self.assertIn("60 seconds", groq_error_message(raised.exception))

    def test_rate_limit_retry_after_header_is_parsed(self):
        response = type("Response", (), {"headers": {"retry-after": "45"}})()
        error = type("RateLimitError", (), {"response": response})()
        self.assertEqual(rate_limit_retry_after(error), 45.0)

    def test_rate_limit_http_date_header_is_parsed(self):
        retry_at = datetime.now(timezone.utc) + timedelta(seconds=45)
        response = type(
            "Response",
            (),
            {"headers": {"Retry-After": format_datetime(retry_at, usegmt=True)}},
        )()
        error = type("RateLimitError", (), {"response": response})()
        self.assertGreaterEqual(rate_limit_retry_after(error), 43)
        self.assertLessEqual(rate_limit_retry_after(error), 45)

    def test_generic_errors_do_not_include_sensitive_exception_text(self):
        error = Exception("request failed with key do-not-print")
        self.assertEqual(
            groq_error_message(error),
            "Groq request failed (Exception). Check the app logs.",
        )

    def test_connection_errors_have_actionable_messages(self):
        request = httpx.Request("GET", "https://api.groq.com/openai/v1/models")
        self.assertIn(
            "internet connection",
            groq_error_message(APIConnectionError(request=request)),
        )
        self.assertIn(
            "timed out",
            groq_error_message(APITimeoutError(request=request)),
        )

    def test_model_not_found_errors_are_actionable(self):
        error = type("NotFoundError", (), {"status_code": 404})()
        self.assertIn("GROQ_MODEL", groq_error_message(error))

    def test_error_messages_never_include_provider_exception_text(self):
        error = Exception("secret key: should-not-appear")
        self.assertNotIn("should-not-appear", groq_error_message(error))

    def test_parses_valid_sentiment_json(self):
        result = parse_groq_sentiment_response(
            '{"overall_sentiment":"BULLISH","sentiment_score":45,'
            '"key_themes":[],"bullish_factors":[],"bearish_factors":[],'
            '"short_term_outlook":"Positive.","risk_factors":[],"confidence":"HIGH"}'
        )
        self.assertEqual(result["overall_sentiment"], "BULLISH")
        self.assertEqual(result["sentiment_score"], 45)

    def test_parses_json_inside_markdown_fences(self):
        content = (
            "```json\n"
            '{"overall_sentiment":"NEUTRAL","sentiment_score":0,'
            '"key_themes":[],"bullish_factors":[],"bearish_factors":[],'
            '"short_term_outlook":"Mixed.","risk_factors":[],"confidence":"LOW"}'
            "\n```"
        )
        self.assertEqual(
            parse_groq_sentiment_response(content)["overall_sentiment"], "NEUTRAL"
        )

    def test_empty_content_has_actionable_error(self):
        with self.assertRaises(GroqInvalidResponseError) as raised:
            parse_groq_sentiment_response("")
        self.assertIn("empty response", groq_error_message(raised.exception))

    def test_truncated_completion_has_actionable_error(self):
        with self.assertRaises(GroqInvalidResponseError) as raised:
            parse_groq_sentiment_response('{"overall_sentiment":', "length")
        self.assertIn("truncated", groq_error_message(raised.exception))

    def test_malformed_json_has_actionable_error(self):
        with self.assertRaises(GroqInvalidResponseError) as raised:
            parse_groq_sentiment_response('{"overall_sentiment":}')
        self.assertIn("malformed or truncated JSON", groq_error_message(raised.exception))

    def test_rejects_missing_or_out_of_range_fields(self):
        with self.assertRaises(GroqInvalidResponseError):
            parse_groq_sentiment_response('{"overall_sentiment":"BULLISH"}')

    def test_news_fetch_retries_after_timeout(self):
        from news_utils import fetch_news_rss

        response = type("Response", (), {"text": "<rss/>", "raise_for_status": lambda self: None})()
        with (
            patch("news_utils.httpx.get", side_effect=[httpx.ReadTimeout("slow"), response]) as get,
            patch("news_utils.time.sleep") as sleep,
        ):
            self.assertEqual(fetch_news_rss("https://news.example/rss"), "<rss/>")
        self.assertEqual(get.call_count, 2)
        sleep.assert_called_once()

    def test_news_fetch_reports_timeout_after_bounded_retries(self):
        from news_utils import NewsFetchTimeoutError, fetch_news_rss

        with (
            patch("news_utils.httpx.get", side_effect=httpx.ReadTimeout("slow")) as get,
            patch("news_utils.time.sleep"),
        ):
            with self.assertRaises(NewsFetchTimeoutError) as raised:
                fetch_news_rss("https://news.example/rss")
        self.assertEqual(get.call_count, 2)
        self.assertEqual(raised.exception.attempts, 2)


if __name__ == "__main__":
    unittest.main()
