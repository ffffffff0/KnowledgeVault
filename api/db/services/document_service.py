import logging
import random
from datetime import datetime

import xxhash
from peewee import fn

from api.db import FileType, StatusEnum, TaskStatus
from api.db.db_models import DB, Document, Knowledgebase, Task, Tenant
from api.db.db_utils import bulk_insert_into_db
from api.db.services.common_service import CommonService
from api.db.services.knowledgebase_service import KnowledgebaseService
from api.utils import current_timestamp, get_format_time, get_uuid
from rag.settings import get_svr_queue_name, SVR_CONSUMER_GROUP_NAME
from rag.utils.redis_conn import REDIS_CONN


class DocumentService(CommonService):
    model = Document

    @classmethod
    @DB.connection_context()
    def get_list(
        cls, kb_id, page_number, items_per_page, orderby, desc, keywords, id, name
    ):
        docs = cls.model.select().where(cls.model.kb_id == kb_id)
        if id:
            docs = docs.where(cls.model.id == id)
        if name:
            docs = docs.where(cls.model.name == name)
        if keywords:
            docs = docs.where(fn.LOWER(cls.model.name).contains(keywords.lower()))
        if desc:
            docs = docs.order_by(cls.model.getter_by(orderby).desc())
        else:
            docs = docs.order_by(cls.model.getter_by(orderby).asc())

        count = docs.count()
        docs = docs.paginate(page_number, items_per_page)
        return list(docs.dicts()), count

    @classmethod
    @DB.connection_context()
    def get_by_kb_id(
        cls,
        kb_id,
        page_number,
        items_per_page,
        orderby,
        desc,
        keywords,
        run_status,
        types,
    ):
        if keywords:
            docs = cls.model.select().where(
                (cls.model.kb_id == kb_id),
                (fn.LOWER(cls.model.name).contains(keywords.lower())),
            )
        else:
            docs = cls.model.select().where(cls.model.kb_id == kb_id)

        if run_status:
            docs = docs.where(cls.model.run.in_(run_status))
        if types:
            docs = docs.where(cls.model.type.in_(types))

        count = docs.count()
        if desc:
            docs = docs.order_by(cls.model.getter_by(orderby).desc())
        else:
            docs = docs.order_by(cls.model.getter_by(orderby).asc())

        if page_number and items_per_page:
            docs = docs.paginate(page_number, items_per_page)

        return list(docs.dicts()), count

    @classmethod
    @DB.connection_context()
    def count_by_kb_id(cls, kb_id, keywords, run_status, types):
        if keywords:
            docs = cls.model.select().where(
                (cls.model.kb_id == kb_id),
                (fn.LOWER(cls.model.name).contains(keywords.lower())),
            )
        else:
            docs = cls.model.select().where(cls.model.kb_id == kb_id)

        if run_status:
            docs = docs.where(cls.model.run.in_(run_status))
        if types:
            docs = docs.where(cls.model.type.in_(types))

        count = docs.count()

        return count

    @classmethod
    @DB.connection_context()
    def get_total_size_by_kb_id(cls, kb_id, keywords="", run_status=[], types=[]):
        query = cls.model.select(fn.COALESCE(fn.SUM(cls.model.size), 0)).where(
            cls.model.kb_id == kb_id
        )

        if keywords:
            query = query.where(fn.LOWER(cls.model.name).contains(keywords.lower()))
        if run_status:
            query = query.where(cls.model.run.in_(run_status))
        if types:
            query = query.where(cls.model.type.in_(types))

        return int(query.scalar()) or 0

    @classmethod
    @DB.connection_context()
    def insert(cls, doc):
        if not cls.save(**doc):
            raise RuntimeError("Database error (Document)!")
        if not KnowledgebaseService.atomic_increase_doc_num_by_id(doc["kb_id"]):
            raise RuntimeError("Database error (Knowledgebase)!")
        return Document(**doc)

    @classmethod
    @DB.connection_context()
    def get_newly_uploaded(cls):
        fields = [
            cls.model.id,
            cls.model.kb_id,
            cls.model.parser_id,
            cls.model.parser_config,
            cls.model.name,
            cls.model.type,
            cls.model.location,
            cls.model.size,
            Knowledgebase.user_id,
            Tenant.embd_id,
            Tenant.img2txt_id,
            Tenant.asr_id,
            cls.model.update_time,
        ]
        docs = (
            cls.model.select(*fields)
            .join(Knowledgebase, on=(cls.model.kb_id == Knowledgebase.id))
            .join(Tenant, on=(Knowledgebase.user_id == Tenant.id))
            .where(
                cls.model.status == StatusEnum.VALID.value,
                ~(cls.model.type == FileType.VIRTUAL.value),
                cls.model.progress == 0,
                cls.model.update_time >= current_timestamp() - 1000 * 600,
                cls.model.run == TaskStatus.RUNNING.value,
            )
            .order_by(cls.model.update_time.asc())
        )
        return list(docs.dicts())

    @classmethod
    @DB.connection_context()
    def get_unfinished_docs(cls):
        fields = [
            cls.model.id,
            cls.model.process_begin_at,
            cls.model.parser_config,
            cls.model.progress_msg,
            cls.model.run,
            cls.model.parser_id,
        ]
        docs = cls.model.select(*fields).where(
            cls.model.status == StatusEnum.VALID.value,
            ~(cls.model.type == FileType.VIRTUAL.value),
            cls.model.progress < 1,
            cls.model.progress > 0,
        )
        return list(docs.dicts())

    @classmethod
    @DB.connection_context()
    def increment_chunk_num(cls, doc_id, kb_id, chunk_num, duration):
        num = (
            cls.model.update(
                chunk_num=cls.model.chunk_num + chunk_num,
                process_duration=cls.model.process_duration + duration,
            )
            .where(cls.model.id == doc_id)
            .execute()
        )
        if num == 0:
            raise LookupError("Document not found which is supposed to be there")
        num = (
            Knowledgebase.update(
                chunk_num=Knowledgebase.chunk_num + chunk_num,
            )
            .where(Knowledgebase.id == kb_id)
            .execute()
        )
        return num

    @classmethod
    @DB.connection_context()
    def decrement_chunk_num(cls, doc_id, kb_id, token_num, chunk_num, duration):
        num = (
            cls.model.update(
                token_num=cls.model.token_num - token_num,
                chunk_num=cls.model.chunk_num - chunk_num,
                process_duration=cls.model.process_duration + duration,
            )
            .where(cls.model.id == doc_id)
            .execute()
        )
        if num == 0:
            raise LookupError("Document not found which is supposed to be there")
        num = (
            Knowledgebase.update(
                token_num=Knowledgebase.token_num - token_num,
                chunk_num=Knowledgebase.chunk_num - chunk_num,
            )
            .where(Knowledgebase.id == kb_id)
            .execute()
        )
        return num

    @classmethod
    @DB.connection_context()
    def clear_chunk_num(cls, doc_id):
        doc = cls.model.get_by_id(doc_id)
        assert doc, "Can't fine document in database."

        num = (
            Knowledgebase.update(
                token_num=Knowledgebase.token_num - doc.token_num,
                chunk_num=Knowledgebase.chunk_num - doc.chunk_num,
                doc_num=Knowledgebase.doc_num - 1,
            )
            .where(Knowledgebase.id == doc.kb_id)
            .execute()
        )
        return num

    @classmethod
    @DB.connection_context()
    def clear_chunk_num_when_rerun(cls, doc_id):
        doc = cls.model.get_by_id(doc_id)
        assert doc, "Can't fine document in database."

        num = (
            Knowledgebase.update(
                token_num=Knowledgebase.token_num - doc.token_num,
                chunk_num=Knowledgebase.chunk_num - doc.chunk_num,
            )
            .where(Knowledgebase.id == doc.kb_id)
            .execute()
        )
        return num

    @classmethod
    @DB.connection_context()
    def get_tenant_id(cls, doc_id):
        docs = (
            cls.model.select(Knowledgebase.user_id)
            .join(Knowledgebase, on=(Knowledgebase.id == cls.model.kb_id))
            .where(
                cls.model.id == doc_id, Knowledgebase.status == StatusEnum.VALID.value
            )
        )
        docs = docs.dicts()
        if not docs:
            return
        return docs[0]["user_id"]

    @classmethod
    @DB.connection_context()
    def get_knowledgebase_id(cls, doc_id):
        docs = cls.model.select(cls.model.kb_id).where(cls.model.id == doc_id)
        docs = docs.dicts()
        if not docs:
            return
        return docs[0]["kb_id"]

    @classmethod
    @DB.connection_context()
    def get_tenant_id_by_name(cls, name):
        docs = (
            cls.model.select(Knowledgebase.user_id)
            .join(Knowledgebase, on=(Knowledgebase.id == cls.model.kb_id))
            .where(
                cls.model.name == name, Knowledgebase.status == StatusEnum.VALID.value
            )
        )
        docs = docs.dicts()
        if not docs:
            return
        return docs[0]["tenant_id"]

    @classmethod
    @DB.connection_context()
    def get_embd_id(cls, doc_id):
        docs = (
            cls.model.select(Knowledgebase.embd_id)
            .join(Knowledgebase, on=(Knowledgebase.id == cls.model.kb_id))
            .where(
                cls.model.id == doc_id, Knowledgebase.status == StatusEnum.VALID.value
            )
        )
        docs = docs.dicts()
        if not docs:
            return
        return docs[0]["embd_id"]

    @classmethod
    @DB.connection_context()
    def get_chunking_config(cls, doc_id):
        configs = (
            cls.model.select(
                cls.model.id,
                cls.model.kb_id,
                cls.model.parser_id,
                cls.model.parser_config,
                Knowledgebase.language,
                Knowledgebase.embd_id,
            )
            .join(Knowledgebase, on=(cls.model.kb_id == Knowledgebase.id))
            .where(cls.model.id == doc_id)
        )
        configs = configs.dicts()
        if not configs:
            return None
        return configs[0]

    @classmethod
    @DB.connection_context()
    def get_doc_id_by_doc_name(cls, doc_name):
        fields = [cls.model.id]
        doc_id = cls.model.select(*fields).where(cls.model.name == doc_name)
        doc_id = doc_id.dicts()
        if not doc_id:
            return
        return doc_id[0]["id"]

    @classmethod
    @DB.connection_context()
    def get_doc_ids_by_doc_names(cls, doc_names):
        if not doc_names:
            return []

        query = cls.model.select(cls.model.id).where(cls.model.name.in_(doc_names))
        return list(query.scalars().iterator())

    @classmethod
    @DB.connection_context()
    def get_thumbnails(cls, docids):
        fields = [cls.model.id, cls.model.kb_id, cls.model.thumbnail]
        return list(cls.model.select(*fields).where(cls.model.id.in_(docids)).dicts())

    @classmethod
    @DB.connection_context()
    def update_parser_config(cls, id, config):
        if not config:
            return
        e, d = cls.get_by_id(id)
        if not e:
            raise LookupError(f"Document({id}) not found.")

        def dfs_update(old, new):
            for k, v in new.items():
                if k not in old:
                    old[k] = v
                    continue
                if isinstance(v, dict):
                    assert isinstance(old[k], dict)
                    dfs_update(old[k], v)
                else:
                    old[k] = v

        dfs_update(d.parser_config, config)
        if not config.get("raptor") and d.parser_config.get("raptor"):
            del d.parser_config["raptor"]
        cls.update_by_id(id, {"parser_config": d.parser_config})

    @classmethod
    @DB.connection_context()
    def get_doc_count(cls, tenant_id):
        docs = (
            cls.model.select(cls.model.id)
            .join(Knowledgebase, on=(Knowledgebase.id == cls.model.kb_id))
            .where(Knowledgebase.user_id == tenant_id)
        )
        return len(docs)

    @classmethod
    @DB.connection_context()
    def begin2parse(cls, docid):
        cls.update_by_id(
            docid,
            {
                "progress": random.random() * 1 / 100.0,
                "progress_msg": "Task is queued...",
                "process_begin_at": get_format_time(),
            },
        )

    @classmethod
    @DB.connection_context()
    def update_meta_fields(cls, doc_id, meta_fields):
        return cls.update_by_id(doc_id, {"meta_fields": meta_fields})

    @classmethod
    @DB.connection_context()
    def update_progress(cls):
        docs = cls.get_unfinished_docs()
        for d in docs:
            try:
                tsks = Task.query(doc_id=d["id"], order_by=Task.create_time)
                if not tsks:
                    continue
                msg = []
                prg = 0
                finished = True
                bad = 0
                has_raptor = False
                has_graphrag = False
                e, doc = DocumentService.get_by_id(d["id"])
                status = doc.run  # TaskStatus.RUNNING.value
                priority = 0
                for t in tsks:
                    if 0 <= t.progress < 1:
                        finished = False
                    if t.progress == -1:
                        bad += 1
                    prg += t.progress if t.progress >= 0 else 0
                    if t.progress_msg.strip():
                        msg.append(t.progress_msg)
                    if t.task_type == "raptor":
                        has_raptor = True
                    elif t.task_type == "graphrag":
                        has_graphrag = True
                    priority = max(priority, t.priority)
                prg /= len(tsks)
                if finished and bad:
                    prg = -1
                    status = TaskStatus.FAIL.value
                elif finished:
                    if (
                        d["parser_config"].get("raptor", {}).get("use_raptor")
                        and not has_raptor
                    ):
                        queue_raptor_o_graphrag_tasks(d, "raptor", priority)
                        prg = 0.98 * len(tsks) / (len(tsks) + 1)
                    elif (
                        d["parser_config"].get("graphrag", {}).get("use_graphrag")
                        and not has_graphrag
                    ):
                        queue_raptor_o_graphrag_tasks(d, "graphrag", priority)
                        prg = 0.98 * len(tsks) / (len(tsks) + 1)
                    else:
                        status = TaskStatus.DONE.value

                msg = "\n".join(sorted(msg))
                info = {
                    "process_duration": datetime.timestamp(datetime.now())
                    - d["process_begin_at"].timestamp(),
                    "run": status,
                }
                if prg != 0:
                    info["progress"] = prg
                if msg:
                    info["progress_msg"] = msg
                else:
                    info["progress_msg"] = (
                        "%d tasks are ahead in the queue..."
                        % get_queue_length(priority)
                    )
                cls.update_by_id(d["id"], info)
            except Exception as e:
                if str(e).find("'0'") < 0:
                    logging.exception("fetch task exception")

    @classmethod
    @DB.connection_context()
    def get_kb_doc_count(cls, kb_id):
        return len(
            cls.model.select(cls.model.id).where(cls.model.kb_id == kb_id).dicts()
        )

    @classmethod
    @DB.connection_context()
    def do_cancel(cls, doc_id):
        try:
            _, doc = DocumentService.get_by_id(doc_id)
            return doc.run == TaskStatus.CANCEL.value or doc.progress < 0
        except Exception:
            pass
        return False


def queue_raptor_o_graphrag_tasks(doc, ty, priority):
    chunking_config = DocumentService.get_chunking_config(doc["id"])
    hasher = xxhash.xxh64()
    for field in sorted(chunking_config.keys()):
        hasher.update(str(chunking_config[field]).encode("utf-8"))

    def new_task():
        nonlocal doc
        return {
            "id": get_uuid(),
            "doc_id": doc["id"],
            "from_page": 100000000,
            "to_page": 100000000,
            "task_type": ty,
            "progress_msg": datetime.now().strftime("%H:%M:%S") + " created task " + ty,
        }

    task = new_task()
    for field in ["doc_id", "from_page", "to_page"]:
        hasher.update(str(task.get(field, "")).encode("utf-8"))
    hasher.update(ty.encode("utf-8"))
    task["digest"] = hasher.hexdigest()
    bulk_insert_into_db(Task, [task], True)
    assert REDIS_CONN.queue_product(
        get_svr_queue_name(priority), message=task
    ), "Can't access Redis. Please check the Redis' status."


def get_queue_length(priority):
    group_info = REDIS_CONN.queue_info(
        get_svr_queue_name(priority), SVR_CONSUMER_GROUP_NAME
    )
    return int(group_info.get("lag", 0))
