from pydantic import BaseModel

class Question(BaseModel):
    message: str
    session_id: str = "default"
    
class Answer(BaseModel):
    message: str