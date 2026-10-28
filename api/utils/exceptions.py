"""
Shared exception types for DocBot service layer.
"""


class StreamInterruptedError(RuntimeError):
    """DOCBOT-2307: the LLM stream failed after tokens reached the client.

    Raised instead of falling back to another provider, which would restart
    the answer and duplicate text. The message is safe to show the user.
    """

    def __init__(self) -> None:
        super().__init__("Answer interrupted, please retry.")


def safe_error_message(error: BaseException | str | None = None) -> str:
    """Sanitize error messages for client responses — never leak exception internals."""
    if isinstance(error, StreamInterruptedError):
        return str(error)
    return "An internal error occurred. Please try again later."


class TokenExpiredError(Exception):
    """
    Raised when a stored entra_interactive access token is past or within 5
    minutes of its expiry. The 'detail' payload is safe to send to the frontend
    — it contains no credentials.
    """

    def __init__(self) -> None:
        self.detail: dict = {
            "error": "token_expired",
            "message": "Azure AD token expired. Please re-authenticate.",
            "requires_reauth": True,
        }
        super().__init__(self.detail["message"])
