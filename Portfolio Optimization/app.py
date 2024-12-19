import json
import logging
from http.client import HTTPException

from celery.bin.control import status
from fastapi import FastAPI, Header, HTTPException
from fastapi.security import HTTPBearer
from firebase_admin import initialize_app, auth
from google.cloud import bigquery, pubsub_v1
from google.oauth2 import service_account
from dotenv import load_dotenv
import os

from pydantic import field_validator, BaseModel
from scripts.regsetup import description
from starlette.middleware.cors import CORSMiddleware

from src.utils.validators import dateUtil

load_dotenv()
API_KEY = os.getenv("API_KEY")
BQ_TABLE = os.getenv("BQ_TABLE")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    initialize_app()
except ValueError:
    pass

security = HTTPBearer()

app = FastAPI(
    title="Portfolio Optimization API",
    description="API for Portfolio Optimization",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*", "http://localhost:5173", "https://portfolio-backend-741957175071.asia-southeast2.run.app",
                   "https://portfolio-frontend-741957175071.asia-southeast2.run.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

publisher = pubsub_v1.PublisherClient()
project_id = os.getenv("PROJECT_ID")
topic_id = os.getenv("TOPIC_ID")
topic_path = publisher.topic_path(project_id, topic_id)


class OptimizationRequest(BaseModel):
    start_date = str
    granularity = str
    user_id = str

    @field_validator('start_date')
    def validate_start_date(cls, value):
        return dateUtil.validate_start_date(value)

    @field_validator('granularity')
    def validate_granularity(cls, value):
        return dateUtil.validate_granularity(value)

    @field_validator('user_id')
    def validate_user_id(cls, value):
        if not value or not isinstance(value, str) or '@' not in value:
            raise ValueError("Invalid user_id format")
        return value


class OptimizationResponse(BaseModel):
    task_id = str
    status = str
    message: str


async def verify_token(x_api_key: str = Header(None, alias="X-API-key")) -> str:
    """Verify the Bearer token from the request header."""
    if x_api_key != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid Token Bearer key"
        )
    return x_api_key

async def verify_firebase_token(authorization: str = Header(None)) -> dict:
    """Verify Firebase ID token from Authorization header."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")

    token = authorization.split("Bearer ")[1]
    try:
        logger.info(f"Attempting to verify token: {token[:20]}...")  # Log first 20 chars of token
        decoded_token = auth.verify_id_token(token)
        logger.info(f"Token verified successfully for UID: {decoded_token.get('uid')}")
        return decoded_token
    except auth.InvalidIdTokenError as e:
        logger.error(f"Invalid ID Token: {str(e)}")
        raise HTTPException(status_code=401, detail="Invalid Firebase token")
    except auth.ExpiredIdTokenError as e:
        logger.error(f"Expired ID Token: {str(e)}")
        raise HTTPException(status_code=401, detail="Firebase token has expired")
    except Exception as e:
        logger.error(f"Unexpected Firebase token verification error: {str(e)}")
        raise HTTPException(status_code=401, detail=f"Token verification failed: {str(e)}")


def publish_optimization_task(request: OptimizationRequest, task_id: str):
    """
    Publish optimization task to Pub/Sub

    Args:
        request (OptimizationRequest): Optimization request details
        task_id (str): Unique task identifier
    """
    try:
        # Prepare message data
        message_data = {
            'task_id': task_id,
            'start_date': request.start_date,
            'granularity': request.granularity,
            'user_id': request.user_id
        }

        # Publish message to Pub/Sub
        message_bytes = json.dumps(message_data).encode('utf-8')
        future = publisher.publish(topic_path, message_bytes)
        message_id = future.result()

        logger.info(f"Published task {task_id} to Pub/Sub. Message ID: {message_id}")
        return message_id
    except Exception as e:
        logger.error(f"Error publishing to Pub/Sub: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Pub/Sub publish error: {str(e)}")

