from pydantic import BaseModel, Field, field_validator

class TenantCreateRequest(BaseModel):
    tenant_id: str = Field(..., description="The ID of the tenant to create.")
    email: str = Field(..., description="The email of the user to invite to the tenant.")
