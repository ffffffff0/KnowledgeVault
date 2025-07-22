from datetime import datetime

from peewee import fn

from api.db import StatusEnum
from api.db.db_models import DB, Search
from api.db.services.common_service import CommonService
from api.utils import current_timestamp, datetime_format


class SearchService(CommonService):
    model = Search

    @classmethod
    def save(cls, **kwargs):
        kwargs["create_time"] = current_timestamp()
        kwargs["create_date"] = datetime_format(datetime.now())
        kwargs["update_time"] = current_timestamp()
        kwargs["update_date"] = datetime_format(datetime.now())
        obj = cls.model.create(**kwargs)
        return obj

    @classmethod
    @DB.connection_context()
    def accessible4deletion(cls, search_id, user_id) -> bool:
        search = (
            cls.model.select(cls.model.id)
            .where(
                cls.model.id == search_id,
                cls.model.created_by == user_id,
                cls.model.status == StatusEnum.VALID.value,
            )
            .first()
        )
        return search is not None
