import json
import os
import trio

import xxhash

from rag.utils.redis_conn import REDIS_CONN

chat_limiter = trio.CapacityLimiter(int(os.environ.get("MAX_CONCURRENT_CHATS", 10)))


def get_llm_cache(llmnm, txt, history, genconf):
    hasher = xxhash.xxh64()
    hasher.update(str(llmnm).encode("utf-8"))
    hasher.update(str(txt).encode("utf-8"))
    hasher.update(str(history).encode("utf-8"))
    hasher.update(str(genconf).encode("utf-8"))

    k = hasher.hexdigest()
    bin = REDIS_CONN.get(k)
    if not bin:
        return
    return bin


def set_llm_cache(llmnm, txt, v, history, genconf):
    hasher = xxhash.xxh64()
    hasher.update(str(llmnm).encode("utf-8"))
    hasher.update(str(txt).encode("utf-8"))
    hasher.update(str(history).encode("utf-8"))
    hasher.update(str(genconf).encode("utf-8"))

    k = hasher.hexdigest()
    REDIS_CONN.set(k, v.encode("utf-8"), 24 * 3600)


def get_tags_from_cache(kb_ids):
    hasher = xxhash.xxh64()
    hasher.update(str(kb_ids).encode("utf-8"))

    k = hasher.hexdigest()
    bin = REDIS_CONN.get(k)
    if not bin:
        return
    return bin


def set_tags_to_cache(kb_ids, tags):
    hasher = xxhash.xxh64()
    hasher.update(str(kb_ids).encode("utf-8"))

    k = hasher.hexdigest()
    REDIS_CONN.set(k, json.dumps(tags).encode("utf-8"), 600)
