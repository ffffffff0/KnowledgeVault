import logging
import base64
import time
from copy import deepcopy

from api.db import LLMType
from api.db.db_models import (
    init_database_tables as init_web_db,
    LLMFactories,
    LLM,
    TenantLLM,
)
from api.db.services.document_service import DocumentService
from api.db.services.knowledgebase_service import KnowledgebaseService
from api.db.services.llm_service import (
    LLMFactoriesService,
    LLMService,
    TenantLLMService,
)
from api import settings


def encode_to_base64(input_string):
    base64_encoded = base64.b64encode(input_string.encode("utf-8"))
    return base64_encoded.decode("utf-8")


def init_llm_factory():
    try:
        LLMService.filter_delete([LLM.fid == "MiniMax" or LLM.fid == "Minimax"])
        LLMService.filter_delete([LLM.fid == "cohere"])
        LLMFactoriesService.filter_delete([LLMFactories.name == "cohere"])
    except Exception:
        pass

    factory_llm_infos = settings.FACTORY_LLM_INFOS
    for factory_llm_info in factory_llm_infos:
        info = deepcopy(factory_llm_info)
        llm_infos = info.pop("llm")
        try:
            LLMFactoriesService.save(**info)
        except Exception:
            pass
        LLMService.filter_delete([LLM.fid == factory_llm_info["name"]])
        for llm_info in llm_infos:
            llm_info["fid"] = factory_llm_info["name"]
            try:
                LLMService.save(**llm_info)
            except Exception:
                pass

    LLMFactoriesService.filter_delete(
        [(LLMFactories.name == "Local") | (LLMFactories.name == "novita.ai")]
    )
    LLMService.filter_delete([LLM.fid == "Local"])
    LLMService.filter_delete([LLM.llm_name == "qwen-vl-max"])
    LLMService.filter_delete([LLM.fid == "Moonshot", LLM.llm_name == "flag-embedding"])
    TenantLLMService.filter_delete(
        [TenantLLM.llm_factory == "Moonshot", TenantLLM.llm_name == "flag-embedding"]
    )
    LLMFactoriesService.filter_delete([LLMFactoriesService.model.name == "QAnything"])
    LLMService.filter_delete([LLMService.model.fid == "QAnything"])
    TenantLLMService.filter_update(
        [TenantLLMService.model.llm_factory == "QAnything"], {"llm_factory": "Youdao"}
    )
    TenantLLMService.filter_update(
        [TenantLLMService.model.llm_factory == "cohere"], {"llm_factory": "Cohere"}
    )
    ## insert openai two embedding models to the current openai user.
    # print("Start to insert 2 OpenAI embedding models...")
    tenant_ids = set([row["tenant_id"] for row in TenantLLMService.get_openai_models()])
    for tid in tenant_ids:
        for row in TenantLLMService.query(llm_factory="OpenAI", tenant_id=tid):
            row = row.to_dict()
            row["model_type"] = LLMType.EMBEDDING.value
            row["llm_name"] = "text-embedding-3-small"
            row["used_tokens"] = 0
            try:
                TenantLLMService.save(**row)
                row = deepcopy(row)
                row["llm_name"] = "text-embedding-3-large"
                TenantLLMService.save(**row)
            except Exception:
                logging.error(
                    f"Failed to insert OpenAI embedding models for tenant {tid}. It may already exist."
                )
                pass
            break
    for kb_id in KnowledgebaseService.get_all_ids():
        KnowledgebaseService.update_document_number_in_init(
            kb_id=kb_id, doc_num=DocumentService.get_kb_doc_count(kb_id)
        )


def init_web_data():
    start_time = time.time()

    init_llm_factory()
    # if not UserService.get_all().count():
    #    init_superuser()

    # add_graph_templates()
    logging.info("init web data success:{}".format(time.time() - start_time))


if __name__ == "__main__":
    init_web_db()
    init_web_data()
