from pydantic import BaseModel
from typing import List, Dict, Union

class DemoResponse(BaseModel):
    summary: List[str]
    interests: List[Dict[str, str]]
    memory: List[Dict[str, Union[str, List[str]]]]
