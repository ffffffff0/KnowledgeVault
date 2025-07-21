from api.utils.log_utils import init_root_logger
init_root_logger("rag_server")

import logging
import os
import signal
import time
import traceback
import uvicorn
import threading
import uuid

from api import settings
from api.apps.main import app
from api import utils
from api.db.services.document_service import DocumentService

from api.db.db_models import init_database_tables as init_web_db
from api.db.init_data import init_web_data
from api.utils import show_configs
from rag.settings import print_rag_settings
from rag.utils.redis_conn import RedisDistributedLock

stop_event = threading.Event()

def update_progress():
    lock_value = str(uuid.uuid4())
    redis_lock = RedisDistributedLock("update_progress", lock_value=lock_value, timeout=60)
    logging.info(f"update_progress lock_value: {lock_value}")
    while not stop_event.is_set():
        try:
            if redis_lock.acquire():
                DocumentService.update_progress()
                redis_lock.release()
            stop_event.wait(6)
        except Exception:
            logging.exception("update_progress exception")
        finally:
            redis_lock.release()


if __name__ == '__main__':
    logging.info(
        f'project base: {utils.file_utils.get_project_base_directory()}'
    )
    show_configs()
    settings.init_settings()
    print_rag_settings()
    # init db
    init_web_db()
    init_web_data()

    def delayed_start_update_progress():
        logging.info("Starting update_progress thread (delayed)")
        t = threading.Thread(target=update_progress, daemon=True)
        t.start()

    threading.Timer(1.0, delayed_start_update_progress).start()
    # start http server
    try:
        logging.info("RAG HTTP server start...")
        uvicorn.run(app=app, host=settings.HOST_IP, port=settings.HOST_PORT)
    except Exception:
        traceback.print_exc()
        time.sleep(1)
        os.kill(os.getpid(), signal.SIGKILL)
