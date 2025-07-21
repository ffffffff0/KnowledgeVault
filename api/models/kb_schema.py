from pydantic import BaseModel, Field, field_validator

class KBCreateRequest(BaseModel):
    name: str
    user_id: str
    embd_id: str
