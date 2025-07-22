import logging
import time
import traceback

from api.db.db_models import close_connection
from api.db.services.task_service import TaskService
from rag.utils.storage_factory import STORAGE_IMPL
from rag.utils.redis_conn import REDIS_CONN


def collect():
    doc_locations = TaskService.get_ongoing_doc_name()
    logging.debug(doc_locations)
    if len(doc_locations) == 0:
        time.sleep(1)
        return
    return doc_locations


def main():
    locations = collect()
    if not locations:
        return
    logging.info(f"TASKS: {len(locations)}")
    for kb_id, loc in locations:
        try:
            if REDIS_CONN.is_alive():
                try:
                    key = "{}/{}".format(kb_id, loc)
                    if REDIS_CONN.exist(key):
                        continue
                    file_bin = STORAGE_IMPL.get(kb_id, loc)
                    REDIS_CONN.transaction(key, file_bin, 12 * 60)
                    logging.info("CACHE: {}".format(loc))
                except Exception as e:
                    traceback.print_stack(e)
        except Exception as e:
            traceback.print_stack(e)


if __name__ == "__main__":
    while True:
        main()
        close_connection()
        time.sleep(1)
