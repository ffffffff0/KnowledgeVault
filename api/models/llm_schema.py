from pydantic import BaseModel, Field, field_validator

class SetLLMRequest(BaseModel):
    llm_factory: str = Field(..., description="The factory of the LLM")
    user_id: str = Field(..., description="The ID of the user")
    api_key: str = Field(..., description="API key for the LLM")

