import time
import logging
import random
from typing import Callable, Any, TypeVar, cast
import requests

logger = logging.getLogger(__name__)
T = TypeVar('T')

def with_retry(
    max_retries: int = 3,
    base_delay: float = 2.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0,
    transient_http_statuses: tuple = (408, 409, 425, 429, 500, 502, 503, 504),
    on_retry: Callable[[Exception, int, float], None] = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator for robust API call retries with exponential backoff and jitter.
    Specifically handles requests exceptions but catches all for safety.
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        def wrapper(*args, **kwargs) -> T:
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    # Determine if we should retry
                    should_retry = False
                    
                    if isinstance(e, requests.HTTPError):
                        status = getattr(getattr(e, "response", None), "status_code", None)
                        if status in transient_http_statuses:
                            should_retry = True
                    elif isinstance(e, (requests.Timeout, requests.ConnectionError)):
                        should_retry = True
                        
                    # If it's a generic exception or we configured to retry anyway, we'll cautiously retry
                    # unless it's the last attempt
                    if attempt >= max_retries - 1 or not should_retry:
                        raise e

                    # Calculate delay: Exponential backoff with Full Jitter
                    exp_delay = min(max_delay, base_delay * (backoff_factor ** attempt))
                    delay = random.uniform(0.5 * exp_delay, exp_delay)
                    
                    if on_retry:
                        on_retry(e, attempt + 1, delay)
                    else:
                        logger.warning(
                            f"[API-RETRY] Attempt {attempt + 1}/{max_retries} failed: {e}. Retrying in {delay:.1f}s"
                        )
                        
                    time.sleep(delay)
            # Should not reach here due to raise e above, but required for typing
            raise Exception("Retry logic failed")
        return cast(Callable[..., T], wrapper)
    return decorator
