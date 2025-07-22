from datetime import datetime

from peewee import fn

from api.db import StatusEnum, TenantPermission
from api.db.db_models import DB, Document, Knowledgebase, Tenant
from api.db.services.common_service import CommonService
from api.utils import current_timestamp, datetime_format


class KnowledgebaseService(CommonService):
    model = Knowledgebase

    @classmethod
    @DB.connection_context()
    def accessible4deletion(cls, kb_id, user_id):
        """Check if a knowledge base can be deleted by a specific user.

        This method verifies whether a user has permission to delete a knowledge base
        by checking if they are the creator of that knowledge base.

        Args:
            kb_id (str): The unique identifier of the knowledge base to check.
            user_id (str): The unique identifier of the user attempting the deletion.

        Returns:
            bool: True if the user has permission to delete the knowledge base,
                  False if the user doesn't have permission or the knowledge base doesn't exist.

        Example:
            >>> KnowledgebaseService.accessible4deletion("kb123", "user456")
            True

        Note:
            - This method only checks creator permissions
            - A return value of False can mean either:
                1. The knowledge base doesn't exist
                2. The user is not the creator of the knowledge base
        """
        # Check if a knowledge base can be deleted by a user
        docs = cls.model.select(
            cls.model.id).where(cls.model.id == kb_id, cls.model.created_by == user_id).paginate(0, 1)
        docs = docs.dicts()
        if not docs:
            return False
        return True

    @classmethod
    @DB.connection_context()
    def is_parsed_done(cls, kb_id):
        # Check if all documents in the knowledge base have completed parsing
        #
        # Args:
        #     kb_id: Knowledge base ID
        #
        # Returns:
        #     If all documents are parsed successfully, returns (True, None)
        #     If any document is not fully parsed, returns (False, error_message)
        from api.db import TaskStatus
        from api.db.services.document_service import DocumentService

        # Get knowledge base information
        kbs = cls.query(id=kb_id)
        if not kbs:
            return False, "Knowledge base not found"
        kb = kbs[0]

        # Get all documents in the knowledge base
        docs, _ = DocumentService.get_by_kb_id(kb_id, 1, 1000, "create_time", True, "", [], [])

        # Check parsing status of each document
        for doc in docs:
            # If document is being parsed, don't allow chat creation
            if doc['run'] == TaskStatus.RUNNING.value or doc['run'] == TaskStatus.CANCEL.value or doc['run'] == TaskStatus.FAIL.value:
                return False, f"Document '{doc['name']}' in dataset '{kb.name}' is still being parsed. Please wait until all documents are parsed before starting a chat."
            # If document is not yet parsed and has no chunks, don't allow chat creation
            if doc['run'] == TaskStatus.UNSTART.value and doc['chunk_num'] == 0:
                return False, f"Document '{doc['name']}' in dataset '{kb.name}' has not been parsed yet. Please parse all documents before starting a chat."

        return True, None

    @classmethod
    @DB.connection_context()
    def list_documents_by_ids(cls, kb_ids):
        # Get document IDs associated with given knowledge base IDs
        # Args:
        #     kb_ids: List of knowledge base IDs
        # Returns:
        #     List of document IDs
        doc_ids = cls.model.select(Document.id.alias("document_id")).join(Document, on=(cls.model.id == Document.kb_id)).where(
            cls.model.id.in_(kb_ids)
        )
        doc_ids = list(doc_ids.dicts())
        doc_ids = [doc["document_id"] for doc in doc_ids]
        return doc_ids

    @classmethod
    @DB.connection_context()
    def get_kb_ids(cls, user_id):
        # Get all knowledge base IDs for a tenant
        # Args:
        #     tenant_id: Tenant ID
        # Returns:
        #     List of knowledge base IDs
        fields = [
            cls.model.id,
        ]
        kbs = cls.model.select(*fields).where(cls.model.user_id == user_id)
        kb_ids = [kb.id for kb in kbs]
        return kb_ids

    @classmethod
    @DB.connection_context()
    def get_detail(cls, kb_id):
        # Get detailed information about a knowledge base
        # Args:
        #     kb_id: Knowledge base ID
        # Returns:
        #     Dictionary containing knowledge base details
        fields = [
            cls.model.id,
            cls.model.embd_id,
            cls.model.avatar,
            cls.model.name,
            cls.model.language,
            cls.model.description,
            cls.model.permission,
            cls.model.doc_num,
            cls.model.token_num,
            cls.model.chunk_num,
            cls.model.parser_id,
            cls.model.parser_config,
            cls.model.pagerank,
            cls.model.create_time,
            cls.model.update_time
            ]
        kbs = cls.model.select(*fields).join(Tenant, on=(
            (Tenant.id == cls.model.user_id) & (Tenant.status == StatusEnum.VALID.value))).where(
            (cls.model.id == kb_id),
            (cls.model.status == StatusEnum.VALID.value)
        )
        if not kbs:
            return
        d = kbs[0].to_dict()
        return d

    @classmethod
    @DB.connection_context()
    def update_parser_config(cls, id, config):
        # Update parser configuration for a knowledge base
        # Args:
        #     id: Knowledge base ID
        #     config: New parser configuration
        e, m = cls.get_by_id(id)
        if not e:
            raise LookupError(f"knowledgebase({id}) not found.")

        def dfs_update(old, new):
            # Deep update of nested configuration
            for k, v in new.items():
                if k not in old:
                    old[k] = v
                    continue
                if isinstance(v, dict):
                    assert isinstance(old[k], dict)
                    dfs_update(old[k], v)
                elif isinstance(v, list):
                    assert isinstance(old[k], list)
                    old[k] = list(set(old[k] + v))
                else:
                    old[k] = v

        dfs_update(m.parser_config, config)
        cls.update_by_id(id, {"parser_config": m.parser_config})

    @classmethod
    @DB.connection_context()
    def delete_field_map(cls, id):
        e, m = cls.get_by_id(id)
        if not e:
            raise LookupError(f"knowledgebase({id}) not found.")

        m.parser_config.pop("field_map", None)
        cls.update_by_id(id, {"parser_config": m.parser_config})

    @classmethod
    @DB.connection_context()
    def get_field_map(cls, ids):
        # Get field mappings for knowledge bases
        # Args:
        #     ids: List of knowledge base IDs
        # Returns:
        #     Dictionary of field mappings
        conf = {}
        for k in cls.get_by_ids(ids):
            if k.parser_config and "field_map" in k.parser_config:
                conf.update(k.parser_config["field_map"])
        return conf

    @classmethod
    @DB.connection_context()
    def get_by_name(cls, kb_name, user_id):
        # Get knowledge base by name and tenant ID
        # Args:
        #     kb_name: Knowledge base name
        #     tenant_id: Tenant ID
        # Returns:
        #     Tuple of (exists, knowledge_base)
        kb = cls.model.select().where(
            (cls.model.name == kb_name)
            & (cls.model.user_id == user_id)
            & (cls.model.status == StatusEnum.VALID.value)
        )
        if kb:
            return True, kb[0]
        return False, None

    @classmethod
    @DB.connection_context()
    def get_all_ids(cls):
        # Get all knowledge base IDs
        # Returns:
        #     List of all knowledge base IDs
        return [m["id"] for m in cls.model.select(cls.model.id).dicts()]

    @classmethod
    @DB.connection_context()
    def get_list(cls, joined_tenant_ids, user_id,
                 page_number, items_per_page, orderby, desc, id, name):
        # Get list of knowledge bases with filtering and pagination
        # Args:
        #     joined_tenant_ids: List of tenant IDs
        #     user_id: Current user ID
        #     page_number: Page number for pagination
        #     items_per_page: Number of items per page
        #     orderby: Field to order by
        #     desc: Boolean indicating descending order
        #     id: Optional ID filter
        #     name: Optional name filter
        # Returns:
        #     List of knowledge bases
        kbs = cls.model.select()
        if id:
            kbs = kbs.where(cls.model.id == id)
        if name:
            kbs = kbs.where(cls.model.name == name)
        kbs = kbs.where(
            ((cls.model.user_id.in_(joined_tenant_ids) & (cls.model.permission ==
                                                            TenantPermission.TEAM.value)) | (
                cls.model.user_id == user_id))
            & (cls.model.status == StatusEnum.VALID.value)
        )
        if desc:
            kbs = kbs.order_by(cls.model.getter_by(orderby).desc())
        else:
            kbs = kbs.order_by(cls.model.getter_by(orderby).asc())

        kbs = kbs.paginate(page_number, items_per_page)

        return list(kbs.dicts())

    @classmethod
    @DB.connection_context()
    def atomic_increase_doc_num_by_id(cls, kb_id):
        data = {}
        data["update_time"] = current_timestamp()
        data["update_date"] = datetime_format(datetime.now())
        data["doc_num"] = cls.model.doc_num + 1
        num = cls.model.update(data).where(cls.model.id == kb_id).execute()
        return num

    @classmethod
    @DB.connection_context()
    def update_document_number_in_init(cls, kb_id, doc_num):
        """
        Only use this function when init system
        """
        ok, kb = cls.get_by_id(kb_id)
        if not ok:
            return
        kb.doc_num = doc_num

        dirty_fields = kb.dirty_fields
        if cls.model._meta.combined.get("update_time") in dirty_fields:
            dirty_fields.remove(cls.model._meta.combined["update_time"])

        if cls.model._meta.combined.get("update_date") in dirty_fields:
            dirty_fields.remove(cls.model._meta.combined["update_date"])

        try:
            kb.save(only=dirty_fields)
        except ValueError as e:
            if str(e) == "no data to save!":
                pass # that's OK
            else:
                raise e

