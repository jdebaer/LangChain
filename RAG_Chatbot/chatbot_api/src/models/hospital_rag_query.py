# We define Pydantic models HospitalQueryInput and HospitalQueryOutput. HospitalQueryInput is used to verify that the POST request body (we only support POST)
# includes a text field, representing the query your chatbot responds to. HospitalQueryOutput verifies the response body sent back to your user includes input, 
# output, and intermediate_step fields.

from pydantic import BaseModel

class HospitalQueryInput(BaseModel):
    text: str

class HospitalQueryOutput(BaseModel):
    input: str
    output: str
    intermediate_steps: list[str]
