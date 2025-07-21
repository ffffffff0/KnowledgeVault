from pydantic import BaseModel, Field, field_validator
from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException, status
from typing import List, Optional


class RunDocument(BaseModel):
    doc_ids: List[str] = Field(..., description="List of document IDs to run")
    user_id: str = Field(..., description="User ID of the requester")
    run: int = Field(..., description="Run identifier for the task")
    

class UpdateParserRequest(BaseModel):
    doc_id: str = Field(..., description="Document ID to update")
    parser_id: str = Field(..., description="New parser ID to set for the document")
    user_id: str = Field(..., description="User ID of the requester")
    parser_config: Optional[dict] = Field(None, description="Optional parser configuration")
