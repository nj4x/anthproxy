import base64
import binascii

from ..mapper import AnthropicRequestError


def extract_aws_credentials(api_key_header: str) -> dict:
    """Extract AWS credentials from a base64-encoded x-api-key header.

    Expected format after decoding: access_key|secret_key|session_token
    """
    api_key = (api_key_header or '').strip()
    if not api_key:
        raise AnthropicRequestError(
            'x-api-key header is required',
            error_type='authentication_error',
            status_code=401,
        )

    try:
        decoded = base64.b64decode(api_key, validate=True).decode('utf-8')
    except (binascii.Error, UnicodeDecodeError, ValueError):
        raise AnthropicRequestError(
            'x-api-key must be a base64-encoded access_key|secret|session_token triplet',
            error_type='authentication_error',
            status_code=401,
        )

    parts = decoded.split('|')
    if len(parts) != 3 or any(not part for part in parts):
        raise AnthropicRequestError(
            'x-api-key must decode to access_key|secret|session_token',
            error_type='authentication_error',
            status_code=401,
        )

    return {
        'aws_access_key_id': parts[0],
        'aws_secret_access_key': parts[1],
        'aws_session_token': parts[2],
    }
