import json
import logging
import uuid
from datetime import datetime
from http.client import HTTPException
from typing import Dict, Any

import granian
import pytz
from fastapi import FastAPI, Header, HTTPException, Depends
from fastapi.security import HTTPBearer
from firebase_admin import initialize_app, auth
from google.cloud import bigquery, pubsub_v1
from google.oauth2 import service_account
from dotenv import load_dotenv
import os

from granian.constants import Interfaces
from pydantic import field_validator, BaseModel
from fastapi.middleware.cors import CORSMiddleware

from src.utils.validators import dateUtil

load_dotenv()
API_KEY = os.getenv("API_KEY")
BQ_TABLE = os.getenv("BQ_TABLE")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')

    # Create credentials object
    credentials = service_account.Credentials.from_service_account_file(credentials_path)
    client = bigquery.Client(credentials=credentials)
except:
    client = bigquery.Client()

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
PROJECT_ID = os.getenv("PROJECT_ID")
TOPIC_ID = os.getenv("TOPIC_ID")
topic_path = publisher.topic_path(PROJECT_ID, TOPIC_ID)


class OptimizationRequest(BaseModel):
    start_date: str
    granularity: str
    user_id: str

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
    task_id: str
    status: str
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


def get_user_portfolio_history(user_id: str) -> Dict[str, Any]:
    """
    Get user's portfolio history from BigQuery
    """
    query = f"""
    SELECT *
    FROM `{BQ_TABLE}`
    WHERE user_id = '{user_id}'
    ORDER BY created_at DESC
    """
    try:
        query_job = client.query(query)
        results = query_job.result()
        portfolio_history = []
        for row in results:
            latest_allocation = row.allocation[-1] if row.allocation else None
            if latest_allocation:
                portfolio_history.append({
                    'generation_id': row.generation_id,
                    'start_date': row.starting_date,
                    'datapoints': row.datapoints,
                    'created_at': row.created_at.strftime('%Y-%m-%d %H:%M:%S'),
                    'optimized_portfolio_month': latest_allocation['period'],
                    'optimized_portfolio_allocation': [
                        {
                            'ticker': alloc['ticker'],
                            'allocation_percentage': alloc['allocation_percentage']
                        }
                        for alloc in latest_allocation['allocation_detail']
                        if alloc['industry'] != 'Benchmark'
                    ]
                })
        return {
            'status': 'success',
            'message': 'Portfolio history retrieved successfully',
            'portfolio_history': portfolio_history
        }
    except Exception as e:
        logger.error(f"Error retrieving results from BigQuery: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving results from BigQuery: {str(e)}"
        )


@app.get("/")
def read_root():
    return {"Status": "OK",
            "Message": "Welcome to Portfolio Optimization API (24-12-19.02)"
            }


@app.post("/optimize", response_model=OptimizationResponse)
def optimize(
        request: OptimizationRequest,
        api_key: str = Depends(verify_token),
        user_claims: dict = Depends(verify_firebase_token)
):
    """Initiate portfolio optimization task.
    1. Validate request data
    2. Generate unique task ID
    3. Publish optimization task to Pub/Sub
    4. Return task ID and status
    """
    try:
        task_id = str(uuid.uuid4())
        publish_optimization_task(request, task_id)

        return {
            "task_id": task_id,
            "status": "Success",
            "message": "Please wait while we optimize your portfolio"
        }
    except Exception as e:
        logger.error(f"Optimization initiation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/users/{user_id}/portfolio-history")
def get_user_portfolios(
        user_id: str,
        api_key: str = Depends(verify_token),
        user_claims: dict = Depends(verify_firebase_token)
) -> Dict[str, Any]:
    """Get user's optimized portfolio history"""
    try:
        logger.info(f"Retrieving portfolio history for user: {user_id}")
        return get_user_portfolio_history(user_id)
    except Exception as e:
        logger.error(f"Error retrieving portfolio history: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/health")
def health_check(
        # api_key: str = Depends(verify_token)
) -> Dict[str, str]:
    """Health check endpoint"""
    wib = pytz.timezone('Asia/Jakarta')
    return {
        'status': 'healthy',
        'timestamp': datetime.now(wib).strftime('%Y-%m-%d %H:%M:%S')
    }


if __name__ == "__main__":
    granian.Granian(
        target="app:app",
        address="0.0.0.0",
        port=8080,
        interface=Interfaces.ASGI,
        workers=4,
        threads=2
    ).serve()
