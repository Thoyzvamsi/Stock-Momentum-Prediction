import time

import httpx


NEWS_REQUEST_TIMEOUT_SECONDS = 10
NEWS_REQUEST_ATTEMPTS = 2
NEWS_RETRY_DELAY_SECONDS = 0.5


class NewsFetchTimeoutError(TimeoutError):
    def __init__(self, attempts: int):
        self.attempts = attempts
        super().__init__(
            f"Google News did not respond after {attempts} attempts."
        )


def fetch_news_rss(url: str) -> str:
    for attempt in range(NEWS_REQUEST_ATTEMPTS):
        try:
            response = httpx.get(
                url,
                timeout=NEWS_REQUEST_TIMEOUT_SECONDS,
                follow_redirects=True,
            )
            response.raise_for_status()
            return response.text
        except httpx.TimeoutException as error:
            if attempt + 1 == NEWS_REQUEST_ATTEMPTS:
                raise NewsFetchTimeoutError(NEWS_REQUEST_ATTEMPTS) from error
            time.sleep(NEWS_RETRY_DELAY_SECONDS)

    raise RuntimeError("News fetch retry loop ended unexpectedly.")
