from fastapi import APIRouter
from api.models.kb_schema import KBCreateRequest
from api.utils.api_utils import server_error_response, get_data_error_result
from api.constants import DATASET_NAME_LIMIT
from api.db.services import duplicate_name
from api.db.services.knowledgebase_service import KnowledgebaseService
from api.db import StatusEnum
from api.utils import get_uuid
from api.utils.api_utils import get_result
from engine.milvus_client import MilvusClientBase


router = APIRouter(tags=["kb"], prefix="/kb")


@router.post("/create")
async def create_kb_handler(req: KBCreateRequest):
    dataset_name = req.name
    if not isinstance(dataset_name, str):
        return get_data_error_result(message="Dataset name must be string.")
    if dataset_name.strip() == "":
        return get_data_error_result(message="Dataset name can't be empty.")
    if len(dataset_name.encode("utf-8")) > DATASET_NAME_LIMIT:
        return get_data_error_result(
            message=f"Dataset name length is {len(dataset_name)} which is larger than {DATASET_NAME_LIMIT}"
        )

    dataset_name = dataset_name.strip()
    dataset_name = duplicate_name(
        KnowledgebaseService.query,
        name=dataset_name,
        user_id=req.user_id,
        status=StatusEnum.VALID.value,
    )
    try:
        kb_info = {**req.dict()}
        kb_info["id"] = get_uuid()
        kb_info["name"] = dataset_name
        kb_info["created_by"] = req.user_id
        kb_info["embd_id"] = req.embd_id or "embedding-001"
        # user default parse config
        kb_info["parser_config"] = {
            "layout_recognize": "DeepDOC",
            "chunk_token_num": 512,
            "delimiter": "\n\n",
            "auto_keywords": 0,
            "auto_questions": 0,
            "html4excel": False,
            "raptor": {
                "use_raptor": False
            },
        }
        kb_info["parser_id"] = "naive"

        print(f"Creating knowledge base with info: {kb_info}")
        if not KnowledgebaseService.save(**kb_info):
            return get_data_error_result()
        # create milvus collection
        MilvusClientBase(user_id=req.user_id, kb_id=kb_info['id'])
        return get_result(data={"kb_id": kb_info["id"]})
    except Exception as e:
        return server_error_response(e)
