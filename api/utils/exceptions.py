"""
Shared exception types for DocBot service layer.
"""


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
