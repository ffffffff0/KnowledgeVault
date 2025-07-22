import functools
import json
import logging
import random
import time
from base64 import b64encode
from copy import deepcopy
from hmac import HMAC
from urllib.parse import quote, urlencode
from uuid import uuid1

import requests
from api import settings
from api.constants import REQUEST_MAX_WAIT_SEC, REQUEST_WAIT_SEC
from api.utils import CustomJSONEncoder

requests.models.complexjson.dumps = functools.partial(json.dumps, cls=CustomJSONEncoder)


def request(**kwargs):
    sess = requests.Session()
    stream = kwargs.pop("stream", sess.stream)
    timeout = kwargs.pop("timeout", None)
    kwargs["headers"] = {
        k.replace("_", "-").upper(): v for k, v in kwargs.get("headers", {}).items()
    }
    prepped = requests.Request(**kwargs).prepare()

    if settings.CLIENT_AUTHENTICATION and settings.HTTP_APP_KEY and settings.SECRET_KEY:
        timestamp = str(round(time() * 1000))
        nonce = str(uuid1())
        signature = b64encode(
            HMAC(
                settings.SECRET_KEY.encode("ascii"),
                b"\n".join(
                    [
                        timestamp.encode("ascii"),
                        nonce.encode("ascii"),
                        settings.HTTP_APP_KEY.encode("ascii"),
                        prepped.path_url.encode("ascii"),
                        prepped.body if kwargs.get("json") else b"",
                        (
                            urlencode(
                                sorted(kwargs["data"].items()),
                                quote_via=quote,
                                safe="-._~",
                            ).encode("ascii")
                            if kwargs.get("data") and isinstance(kwargs["data"], dict)
                            else b""
                        ),
                    ]
                ),
                "sha1",
            ).digest()
        ).decode("ascii")

        prepped.headers.update(
            {
                "TIMESTAMP": timestamp,
                "NONCE": nonce,
                "APP-KEY": settings.HTTP_APP_KEY,
                "SIGNATURE": signature,
            }
        )

    return sess.send(prepped, stream=stream, timeout=timeout)


def get_exponential_backoff_interval(retries, full_jitter=False):
    """Calculate the exponential backoff wait time."""
    # Will be zero if factor equals 0
    countdown = min(REQUEST_MAX_WAIT_SEC, REQUEST_WAIT_SEC * (2**retries))
    # Full jitter according to
    # https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter/
    if full_jitter:
        countdown = random.randrange(countdown + 1)
    # Adjust according to maximum wait time and account for negative values.
    return max(0, countdown)


def get_data_error_result(
    code=settings.RetCode.DATA_ERROR, message="Sorry! Data missing!"
):
    logging.exception(Exception(message))
    result_dict = {"code": code, "message": message}
    response = {}
    for key, value in result_dict.items():
        if value is None and key != "code":
            continue
        else:
            response[key] = value
    return response


def server_error_response(e):
    logging.exception(e)
    try:
        if e.code == 401:
            return get_result(code=401, message=repr(e))
    except BaseException:
        pass
    if len(e.args) > 1:
        return get_result(
            code=settings.RetCode.EXCEPTION_ERROR,
            message=repr(e.args[0]),
            data=e.args[1],
        )
    if repr(e).find("index_not_found_exception") >= 0:
        return get_result(
            code=settings.RetCode.EXCEPTION_ERROR,
            message="No chunk found, please upload file and parse it.",
        )

    return get_result(code=settings.RetCode.EXCEPTION_ERROR, message=repr(e))


def is_localhost(ip):
    return ip in {"127.0.0.1", "::1", "[::1]", "localhost"}


def get_result(code=settings.RetCode.SUCCESS, message="success", data=None):
    return {"code": code, "message": message, "data": data}


def get_result(code=settings.RetCode.SUCCESS, message="", data=None):
    if code == 0:
        if data is not None:
            response = {"code": code, "data": data}
        else:
            response = {"code": code}
    else:
        response = {"code": code, "message": message}
    return response


def get_error_argument_result(message="Invalid arguments"):
    return get_result(code=settings.RetCode.ARGUMENT_ERROR, message=message)


def get_error_permission_result(message="Permission error"):
    return get_result(code=settings.RetCode.PERMISSION_ERROR, message=message)


def get_error_operating_result(message="Operating error"):
    return get_result(code=settings.RetCode.OPERATING_ERROR, message=message)


def get_parser_config(chunk_method, parser_config):
    if parser_config:
        return parser_config
    if not chunk_method:
        chunk_method = "naive"
    key_mapping = {
        "naive": {
            "chunk_token_num": 128,
            "delimiter": r"\n",
            "html4excel": False,
            "layout_recognize": "DeepDOC",
            "raptor": {"use_raptor": False},
        },
        "qa": {"raptor": {"use_raptor": False}},
        "tag": None,
        "resume": None,
        "manual": {"raptor": {"use_raptor": False}},
        "table": None,
        "paper": {"raptor": {"use_raptor": False}},
        "book": {"raptor": {"use_raptor": False}},
        "laws": {"raptor": {"use_raptor": False}},
        "presentation": {"raptor": {"use_raptor": False}},
        "one": None,
        "knowledge_graph": {
            "chunk_token_num": 8192,
            "delimiter": r"\n",
            "entity_types": ["organization", "person", "location", "event", "time"],
        },
        "email": None,
        "picture": None,
    }
    parser_config = key_mapping[chunk_method]
    return parser_config


def get_data_openai(
    id=None,
    created=None,
    model=None,
    prompt_tokens=0,
    completion_tokens=0,
    content=None,
    finish_reason=None,
    object="chat.completion",
    param=None,
):
    total_tokens = prompt_tokens + completion_tokens
    return {
        "id": f"{id}",
        "object": object,
        "created": int(time.time()) if created else None,
        "model": model,
        "param": param,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "completion_tokens_details": {
                "reasoning_tokens": 0,
                "accepted_prediction_tokens": 0,
                "rejected_prediction_tokens": 0,
            },
        },
        "choices": [
            {
                "message": {"role": "assistant", "content": content},
                "logprobs": None,
                "finish_reason": finish_reason,
                "index": 0,
            }
        ],
    }


def check_duplicate_ids(ids, id_type="item"):
    """
    Check for duplicate IDs in a list and return unique IDs and error messages.

    Args:
        ids (list): List of IDs to check for duplicates
        id_type (str): Type of ID for error messages (e.g., 'document', 'dataset', 'chunk')

    Returns:
        tuple: (unique_ids, error_messages)
            - unique_ids (list): List of unique IDs
            - error_messages (list): List of error messages for duplicate IDs
    """
    id_count = {}
    duplicate_messages = []

    # Count occurrences of each ID
    for id_value in ids:
        id_count[id_value] = id_count.get(id_value, 0) + 1

    # Check for duplicates
    for id_value, count in id_count.items():
        if count > 1:
            duplicate_messages.append(f"Duplicate {id_type} ids: {id_value}")

    # Return unique IDs and error messages
    return list(set(ids)), duplicate_messages


def deep_merge(default: dict, custom: dict) -> dict:
    """
    Recursively merges two dictionaries with priority given to `custom` values.

    Creates a deep copy of the `default` dictionary and iteratively merges nested
    dictionaries using a stack-based approach. Non-dict values in `custom` will
    completely override corresponding entries in `default`.

    Args:
        default (dict): Base dictionary containing default values.
        custom (dict): Dictionary containing overriding values.

    Returns:
        dict: New merged dictionary combining values from both inputs.

    Example:
        >>> from copy import deepcopy
        >>> default = {"a": 1, "nested": {"x": 10, "y": 20}}
        >>> custom = {"b": 2, "nested": {"y": 99, "z": 30}}
        >>> deep_merge(default, custom)
        {'a': 1, 'b': 2, 'nested': {'x': 10, 'y': 99, 'z': 30}}

        >>> deep_merge({"config": {"mode": "auto"}}, {"config": "manual"})
        {'config': 'manual'}

    Notes:
        1. Merge priority is always given to `custom` values at all nesting levels
        2. Non-dict values (e.g. list, str) in `custom` will replace entire values
           in `default`, even if the original value was a dictionary
        3. Time complexity: O(N) where N is total key-value pairs in `custom`
        4. Recommended for configuration merging and nested data updates
    """
    merged = deepcopy(default)
    stack = [(merged, custom)]

    while stack:
        base_dict, override_dict = stack.pop()

        for key, val in override_dict.items():
            if (
                key in base_dict
                and isinstance(val, dict)
                and isinstance(base_dict[key], dict)
            ):
                stack.append((base_dict[key], val))
            else:
                base_dict[key] = val

    return merged


def remap_dictionary_keys(source_data: dict, key_aliases: dict = None) -> dict:
    """
    Transform dictionary keys using a configurable mapping schema.

    Args:
        source_data: Original dictionary to process
        key_aliases: Custom key transformation rules (Optional)
            When provided, overrides default key mapping
            Format: {<original_key>: <new_key>, ...}

    Returns:
        dict: New dictionary with transformed keys preserving original values

    Example:
        >>> input_data = {"old_key": "value", "another_field": 42}
        >>> remap_dictionary_keys(input_data, {"old_key": "new_key"})
        {'new_key': 'value', 'another_field': 42}
    """
    DEFAULT_KEY_MAP = {
        "chunk_num": "chunk_count",
        "doc_num": "document_count",
        "parser_id": "chunk_method",
        "embd_id": "embedding_model",
    }

    transformed_data = {}
    mapping = key_aliases or DEFAULT_KEY_MAP

    for original_key, value in source_data.items():
        mapped_key = mapping.get(original_key, original_key)
        transformed_data[mapped_key] = value

    return transformed_data
