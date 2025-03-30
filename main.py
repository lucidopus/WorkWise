import uvicorn, os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.requests import Request

from utils.config import API_NAME, origins
from utils.helper import pipeline


app = FastAPI(
    title=API_NAME,
    description="API for WorkWise AI.",
    version="0.1.0",
    openapi_tags=[
        {"name": API_NAME, "description": "Endpoints for WorkWise AI"},
    ],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Index"], response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/robots.txt", include_in_schema=False)
async def get_robots_txt():
    robots_txt_path = os.path.join("static", "robots.txt")
    return FileResponse(robots_txt_path, media_type="text/plain")


templates = Jinja2Templates(directory="static")
from pydantic import BaseModel

class UserInput(BaseModel):
    user_message: str

@app.post(
    path="/get_response",
    tags=["Model Endpoints"],
    response_description="Successful Response",
    description="Get WorkWise AI's response based on the conversation history.",
    name=API_NAME,
)
async def process(input_data: UserInput):
    user_message = input_data.user_message
    print("User message received:", user_message)

    response = pipeline(user_message)
    
    if 'intermediate_steps' in response.keys():
        print(response)
        if len(response["intermediate_steps"]) > 0:
            if response['intermediate_steps'][0][0].tool == 'get_job_related_data':
                plot = [response['intermediate_steps'][0][1].to_json()]
            elif response['intermediate_steps'][0][0].tool == 'tavily_search':
                plot = []
            elif response['intermediate_steps'][0][0].tool == 'company_info':
                plot = []
            elif response['intermediate_steps'][0][0].tool == 'salary_info':
                plot = [response['intermediate_steps'][0][1].to_json()]
        else:
            plot = []
        
    else:
        plot = []

    return {
        "response": response['output'],
        "plot": plot,  
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True, workers=4)

