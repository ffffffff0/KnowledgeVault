from api.utils.log_utils import init_root_logger
init_root_logger("milvus_client")

import copy
import logging

from pymilvus import (
    connections,
    DataType,
    Partition,
    MilvusClient,
    Function,
    FunctionType,
)
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import List
from rag import settings

ATTEMPT_TIME = 2

class MilvusClientBase:
    def __init__(self, user_id, kb_id, *, client_timeout=3):
        self.user_id = user_id
        self.kb_id = kb_id
        self.kb_name = f"kb_{kb_id}"
        self.host = settings.MILVUS["host"]
        self.port = settings.MILVUS["port"]
        self.vector_dim = 768
        # self.user = MILVUS_USER
        # self.password = MILVUS_PASSWORD
        self.db_name = user_id
        self.client_timeout = client_timeout
        self.sess = None
        self.partitions: List[Partition] = []
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.top_k = settings.MILVUS["topk"]
        self.search_params = {"metric_type": "L2", "params": {"nprobe": 256}}
        self.init()

    @property
    def output_fields(self):
        return [
            "chunk_id",
            "kb_id",
            "user_id",
            "create_at",
        ]

    def init(self):
        try:
            # connections.connect(host=self.host, port=self.port, user=self.user,
            #                     password=self.password, db_name=self.db_name)
            connections.connect(host=self.host, port=self.port)
            milvus_client = MilvusClient(uri=f"http://{self.host}:{self.port}")
            existing_databases = milvus_client.list_databases()

            if self.db_name in existing_databases:
                print(f"✓ Database '{self.db_name}' already exists.")
            else:
                print(f"✗ Database '{self.db_name}' does not exist, creating it now...")
                milvus_client.create_database(self.db_name)
                print(f"Database '{self.db_name}' created successfully.")

            milvus_client.using_database(self.db_name)
            print(f"switched to db: '{self.db_name}'")
            self.sess = milvus_client
        except Exception as e:
            print(f"Failed to initialize Milvus client: {e}")
            raise e

    def create_collection(self, vector_size: int):
        try:
            if not self.sess.has_collection(self.kb_name):
                schema = self.sess.create_schema()
                schema.add_field(field_name="chunk_id", datatype=DataType.VARCHAR, max_length=64, is_primary=True)
                schema.add_field(field_name="doc_id", datatype=DataType.VARCHAR, max_length=64)
                schema.add_field(field_name="user_id", datatype=DataType.VARCHAR, max_length=64)
                schema.add_field(field_name="kb_id", datatype=DataType.VARCHAR, max_length=64)
                schema.add_field(field_name="create_at", datatype=DataType.VARCHAR, max_length=64)
                schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=4000, enable_analyzer=True)
                schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=vector_size)
                schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
                bm25_function = Function(
                    name="text_bm25_emb",
                    input_field_names=["content"],
                    output_field_names=["sparse_vector"],
                    function_type=FunctionType.BM25
                )

                schema.add_function(bm25_function)
                index_params = self.sess.prepare_index_params()
                index_params.add_index(
                    field_name="sparse_vector",
                    index_type="SPARSE_INVERTED_INDEX",
                    metric_type="BM25",
                    params={
                        "inverted_index_algo": "DAAT_MAXSCORE",
                        "bm25_k1": 1.2,
                        "bm25_b": 0.75
                    }
                )
                index_params.add_index(
                    field_name="dense_vector",
                    index_type="HNSW",
                    metric_type="IP",
                    params={
                        "efConstruction": 128,
                        "M": 16
                    }
                )
                self.sess.create_collection(
                    collection_name=self.kb_name,
                    schema=schema,
                    index_params=index_params,
                )
        except Exception as e:
            print(f"Failed to create collection: {e}")
            raise e

    def insert(self, documents: list[dict],  user_id: str, knowledgebaseId: str = None) -> list[str]:
        data = []
        for d in documents:
            assert "_id" not in d
            assert "id" in d
            d_copy = copy.deepcopy(d)
            data.append(
                {
                    "chunk_id": d_copy.get("id", ""),
                    "doc_id": d_copy.get("doc_id", ""),
                    "user_id": user_id,
                    "kb_id": knowledgebaseId,
                    "create_at": d_copy.get("create_at", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                    "content": d_copy.get("content_with_weight", ""),
                    "dense_vector": d_copy.get("dense_vector", []),
                }
            )

        res = []
        for _ in range(ATTEMPT_TIME):
            try:
                self.sess.insert(collection_name=self.kb_name, data=data)
            except Exception as e:
                logging.error(f"Failed to insert data into Milvus: {e}")
                res.append(str(e))

        return res
    
    def delete(self, condition: str) -> int:
        for _ in range(ATTEMPT_TIME):
            try:
                self.sess.delete(
                    collection_name=self.kb_name,
                    filter=condition,
                )
                return 1
            except Exception as e:
                logging.error(f"Failed to delete data from Milvus: {e}")
                return 0
        return 0
    
    def filter(self, condition: str) -> list[dict]:
        for _ in range(ATTEMPT_TIME):
            try:
                res = self.sess.query(
                    collection_name=self.kb_name,
                    filter=condition,
                    output_fields=self.output_fields,
                )
                return res
            except Exception as e:
                logging.error(f"Failed to filter data from Milvus: {e}")
                return []

        return []
