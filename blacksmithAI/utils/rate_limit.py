"""
Rate limit handling utilities for LLM API calls with exponential backoff.
"""
import asyncio
import random
import logging
from functools import wraps
from typing import Callable, Any

logger = logging.getLogger(__name__)
def is_rate_limit_exception(exc: Exception) -> bool:
    """
    Check if exception is a rate limit error (HTTP 429 or 503).

    Handles both:
    - requests.Response objects with status_code attribute
    - Custom exceptions with string representation containing rate limit info
    """
    err_str = str(exc)

    # Check for HTTP 429 or 503 status codes
    if hasattr(exc, 'status_code'):
        return exc.status_code in (429, 503, 504)

    # Check for common rate limit error patterns
    rate_limit_patterns = [
        '429',
        'rate limit',
        'temporarily rate-limited',
        'upstream',
        '503',
        '504'
    ]

    return any(pattern.lower() in err_str.lower() for pattern in rate_limit_patterns)
async def exponential_backoff_with_jitter(
    attempt: int,
    min_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0
) -> float:
    """
    Calculate exponential backoff delay with jitter to prevent thundering herd.

    Args:
        attempt: Current retry attempt (0-indexed)
        min_delay: Minimum delay in seconds
        max_delay: Maximum delay in seconds
        backoff_factor: Factor by which delay increases each retry

    Returns:
        Calculated delay with random jitter applied
    """
    # Exponential backoff: min_delay * (backoff_factor ^ attempt)
    delay = min(min_delay * (backoff_factor ** attempt), max_delay)

    # Add jitter: random value between 0.5x and 1.5x the calculated delay
    jitter = random.uniform(0.5, 1.5)
    return delay * jitter
async def robust_async_call(
    func: Callable,
    *args,
    max_retries: int = 5,
    initial_delay: float = 1.0,
    **kwargs
) -> Any:
    """
    Robust async call wrapper with exponential backoff for rate limits.

    This decorator wraps async functions (particularly LLM API calls) to handle
    transient errors like rate limits (429) and server errors (503, 504) with
    exponential backoff and jitter.

    Args:
        func: Async function to wrap
        max_retries: Maximum number of retry attempts (default: 5)
        initial_delay: Initial delay before first retry in seconds (default: 1.0)
        *args, **kwargs: Arguments to pass to the wrapped function

    Returns:
        Function result on success

    Raises:
        RateLimitError: If all retry attempts are exhausted
        Exception: Re-raises non-rate-limit exceptions immediately
    """
    last_exception = None
    delay = initial_delay

    for attempt in range(max_retries + 1):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            # Check if this is a rate limit or transient error
            if is_rate_limit_exception(e) and attempt < max_retries:
                # Calculate backoff with jitter
                wait_time = await exponential_backoff_with_jitter(
                    attempt, initial_delay
                )

                logger.warning(
                    f"Rate limit or transient error in {func.__name__}. "
                    f"Retrying in {wait_time:.1f}s (attempt {attempt + 1}/{max_retries})"
                )

                # Async sleep to avoid blocking event loop
                await asyncio.sleep(wait_time)

                # Increase delay for next retry
                delay *= backoff_factor
            else:
                # Non-retryable error or max retries exceeded
                last_exception = e
                break

    # Raise the last exception if we exhausted retries
    if last_exception:
        raise RateLimitError(
            f"All {max_retries} retry attempts exhausted for {func.__name__}: "
            f"{last_exception}"
        ) from last_exception

    # This should not be reached, but included for safety
    raise RateLimitError(
        f"Unknown rate limit error in {func.__name__}"
    )
class RateLimitError(Exception):
    """
    Custom exception for rate limit and transient errors.

    Raised when all retry attempts are exhausted.
    """
    pass
def handle_rate_limits_sync(max_retries: int = 3, initial_delay: float = 1.0):
    """
    Decorator for synchronous functions that need rate limit handling.

    Note: For async code, use robust_async_call directly.

    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if is_rate_limit_exception(e) and attempt < max_retries:
                        logger.warning(
                            f"Rate limit in {func.__name__}. "
                            f"Retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries})"
                        )
                        time.sleep(delay)
                        delay *= 2  # Exponential backoff
                    else:
                        raise
            raise RateLimitError(f"Max retries exceeded in {func.__name__}")
        return wrapper
    return decorator
# Pre-configured async retry function for common LLM calls
async def call_llm_with_retry(llm_callable, messages, **llm_kwargs):
    """
    Convenience function for calling LLMs with built-in retry logic.

    Args:
        llm_callable: The LLM call method (e.g., model.invoke)
        messages: Messages to send to the LLM
        **llm_kwargs: Additional keyword arguments for the LLM call

    Returns:
        LLM response
    """
    return await robust_async_call(
        llm_callable,
        messages,
        max_retries=5,
        initial_delay=1.0
    )
# Utility function to check if a response object indicates rate limiting
def check_response_for_rate_limit(response, status_code_key: str = "status_code") -> bool:
    """
    Check if a response object indicates rate limiting.

    Args:
        response: Response object to check
        status_code_key: Key to access status code in dict responses

    Returns:
        True if response indicates rate limiting
    """
    # For requests.Response objects
    if hasattr(response, 'status_code'):
        return response.status_code in (429, 503, 504)

    # For dict responses
    if isinstance(response, dict):
        status = response.get(status_code_key)
        return status in (429, 503, 504)

    return False