import settings
import re
from fastapi import APIRouter, UploadFile, File, Form
from typing import List
from api.constants import FILE_NAME_LEN_LIMIT
from api.db.services.knowledgebase_service import KnowledgebaseService
from api.db.services.file_service import FileService
from api.db.services.document_service import DocumentService
from api.db.services.file2document_service import File2DocumentService
from api.utils.api_utils import get_result, get_data_error_result, server_error_response
from api.models.document_schema import RunDocument, UpdateParserRequest
from api.db.services.task_service import queue_tasks
from api.db import (
    FileType,
    ParserType,
    TaskStatus,
)
from api.db.db_models import File, Task
from engine.milvus_client import MilvusClientBase


router = APIRouter(tags=["document"], prefix="/document")


@router.post("/upload")
async def upload_document(
    kb_id: str = Form(..., description="知识库ID"),
    files: List[UploadFile] = File(..., description="要上传的文件列表"),
    user_id: str = Form(..., description="用户ID"),
):
    if not kb_id:
        return get_result(
            data=False, message='Lack of "KB ID"', code=settings.RetCode.ARGUMENT_ERROR
        )

    for file_obj in files:
        if file_obj.filename == "":
            return get_result(
                data=False,
                message="No file selected!",
                code=settings.RetCode.ARGUMENT_ERROR,
            )
        if len(file_obj.filename.encode("utf-8")) > FILE_NAME_LEN_LIMIT:
            return get_result(
                data=False,
                message=f"File name must be {FILE_NAME_LEN_LIMIT} bytes or less.",
                code=settings.RetCode.ARGUMENT_ERROR,
            )

    e, kb = KnowledgebaseService.get_by_id(kb_id)
    if not e:
        raise LookupError("Can't find this knowledgebase!")
    err, files = await FileService.upload_document(kb, files, user_id)

    if err:
        return get_result(
            data="error", message="\n".join(err), code=settings.RetCode.SERVER_ERROR
        )

    if not files:
        return get_result(
            data=files,
            message="There seems to be an issue with your file format. Please verify it is correct and not corrupted.",
            code=settings.RetCode.DATA_ERROR,
        )

    files = [f[0] for f in files]
    return get_result(data=files)


@router.get("/list")
async def list_docs(
    kb_id: str = None,
    page: int = 0,
    page_size: int = 10,
    desc: bool = True,
    orderby: str = "create_time",
):
    if not kb_id:
        return get_result(
            data=False, message='Lack of "KB ID"', code=settings.RetCode.ARGUMENT_ERROR
        )

    try:
        docs, tol = DocumentService.get_by_kb_id(
            kb_id, page, page_size, orderby, desc, "", [], []
        )
        return get_result(data={"total": tol, "docs": docs})
    except Exception as e:
        return server_error_response(e)


@router.post("/change_parser")
async def change_parser(req: UpdateParserRequest):
    try:
        e, doc = DocumentService.get_by_id(req.doc_id)
        if not e:
            return get_data_error_result(message="Document not found!")
        if doc.parser_id.lower() == req.parser_id.lower():
            if "parser_config" in req:
                if req.parser_config == doc.parser_config:
                    return get_result(data=True)
            else:
                return get_result(data=True)

        if (doc.type == FileType.VISUAL and req.parser_id != "picture") or (
            re.search(r"\.(ppt|pptx|pages)$", doc.name)
            and req.parser_id != "presentation"
        ):
            return get_data_error_result(message="Not supported yet!")

        e = DocumentService.update_by_id(
            doc.id,
            {
                "parser_id": req.parser_id,
                "progress": 0,
                "progress_msg": "",
                "run": TaskStatus.UNSTART.value,
            },
        )
        if not e:
            return get_data_error_result(message="Document not found!")
        if "parser_config" in req:
            DocumentService.update_parser_config(doc.id, req.parser_config)

        e = DocumentService.increment_chunk_num(
            doc.id,
            doc.kb_id,
            doc.chunk_num * -1,
            doc.process_duration * -1,
        )
        if not e:
            return get_data_error_result(message="Document not found!")
        # delete the document from the milvus db
        mc = MilvusClientBase(user_id=req.user_id, kb_id=doc.kb_id)
        exist_filter = f"kb_id == {doc.kb_id} AND doc_id == {doc.id}"
        filter_res = mc.filter(exist_filter)
        if filter_res:
            delete_filter = f"kb_id == {doc.kb_id} AND doc_id == {doc.id}"
            mc.delete(delete_filter)

        return get_result(data=True)
    except Exception as e:
        return server_error_response(e)


@router.post("/run")
async def run_document(req: RunDocument):
    try:
        kb_table_num_map = {}
        for id in req.doc_ids:
            info = {"run": str(req.run), "progress": 0}
            if str(req.run) == TaskStatus.RUNNING.value:
                info["progress_msg"] = ""
                info["chunk_num"] = 0

            e, doc = DocumentService.get_by_id(id)
            if not e:
                return get_data_error_result(message="Document not found!")
            if doc.run == TaskStatus.DONE.value:
                DocumentService.clear_chunk_num_when_rerun(doc.id)

            DocumentService.update_by_id(id, info)
            user_id = DocumentService.get_tenant_id(id)
            if not user_id:
                return get_data_error_result(message="Tenant not found!")
            e, doc = DocumentService.get_by_id(id)
            if not e:
                return get_data_error_result(message="Document not found!")

            e, doc = DocumentService.get_by_id(id)
            doc = doc.to_dict()
            doc["user_id"] = user_id

            doc_parser = doc.get("parser_id", ParserType.NAIVE)
            if doc_parser == ParserType.TABLE:
                kb_id = doc.get("kb_id")
                if not kb_id:
                    continue
                if kb_id not in kb_table_num_map:
                    count = DocumentService.count_by_kb_id(
                        kb_id=kb_id,
                        keywords="",
                        run_status=[TaskStatus.DONE],
                        types=[],
                    )
                    kb_table_num_map[kb_id] = count
                    if kb_table_num_map[kb_id] <= 0:
                        KnowledgebaseService.delete_field_map(kb_id)
            bucket, name = File2DocumentService.get_storage_address(doc_id=doc["id"])
            queue_tasks(doc, bucket, name, 0)

        return get_result(data=True)
    except Exception as e:
        return server_error_response(e)
