import json

from google.cloud import bigquery, pubsub_v1
from dotenv import load_dotenv
from google.oauth2 import service_account
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer
from fastapi.responses import JSONResponse
from granian.constants import Interfaces
from pydantic import BaseModel, field_validator
from typing import Dict, Any, List, Optional
from firebase_admin import auth, initialize_app
import uuid
import os
import pytz
import granian
import logging

from src.optimizer import run_optimization_pipeline
from src.utils.validators import *

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

API_KEY = os.getenv("API_KEY")
PROJECT_ID = os.getenv("PROJECT_ID")
PUBSUB_TOPIC = os.getenv("PUBSUB_TOPIC")
BQ_TABLE = os.getenv("BQ_TABLE")
try:
    credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')

    credentials = service_account.Credentials.from_service_account_file(credentials_path)
    client = bigquery.Client(credentials=credentials)
except:
    client = bigquery.Client()

try:
    initialize_app()
except ValueError:
    pass

security = HTTPBearer()

# Initialize FastAPI app
app = FastAPI(
    title="Portfolio Optimization API",
    description="API for portfolio optimization and analysis",
    version="2.0.1",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*", "http://localhost:5173", "https://portfolio-backend-741957175071.asia-southeast2.run.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

optimization_results = {}


class OptimizationRequest(BaseModel):
    start_date: str
    granularity: int
    user_id: str

    @field_validator('start_date')
    def validate_start_date(cls, v):
        return dateUtil.validate_start_date(v)

    @field_validator('granularity')
    def validate_granularity(cls, v):
        return dateUtil.validate_granularity(v)

    @field_validator('user_id')
    def validate_user_id(cls, v):
        if not v or not isinstance(v, str) or '@' not in v:
            raise ValueError("Invalid user_id format")
        return v


class OptimizationRequestV2(OptimizationRequest):
    volatility: float
    sector_limits: Optional[Dict[str, float]] = None

    @field_validator('volatility')
    def validate_volatility(cls, v):
        if not 0 < v < 1:
            raise ValueError("Volatility must be between 0 and 1")
        return v

    @field_validator('sector_limits')
    def validate_sector_limits(cls, v):
        if v is not None:
            required_sectors = {'Financial', 'Technology', 'Energy', 'Healthcare', 'Industrial', 'Materials', 'Other'}
            if not all(sector in v for sector in required_sectors):
                raise ValueError(f"Missing required sectors. Required: {required_sectors}")
            if not all(0 <= limit <= 1 for limit in v.values()):
                raise ValueError("Sector limits must be between 0 and 1")
        return v


class PortfolioAllocation(BaseModel):
    ticker: str
    allocation_percentage: float


class OptimizationResponse(BaseModel):
    task_id: str
    status: str
    message: str


class OptimizationResult(BaseModel):
    status: str
    message: str
    portfolio_recommendation: Optional[List[PortfolioAllocation]] = None
    metadata: Optional[Dict[str, Any]] = None


def get_latest_allocation(generation_id: str) -> Dict[str, Any]:
    """
    Retrieve the most recent portfolio allocation for a specific generation ID from BigQuery.

    Args:
        generation_id (str): Unique identifier for the optimization generation

    Returns:
        Dict[str, Any]: Portfolio allocation data containing:
            - status (str): Operation status ('success' or 'error')
            - message (str): Status description
            - portfolio_recommendation (List[Dict]): List of portfolio allocations
                - ticker (str): Stock ticker
                - allocation_percentage (float): Percentage allocation
            - metadata (Dict): Additional information including:
                - generation_id (str): Optimization generation ID
                - user_id (str): User identifier
                - start_date (str): Start date of optimization
                - datapoints (int): Number of data points used
                - created_at (str): Timestamp of creation

    Raises:
        HTTPException: If query fails or no data is found
            - 404: If no allocation found for generation_id
            - 500: If database query fails
    """
    query = f"""
    WITH latest_allocation AS (
        SELECT 
            generation_id,
            user_id,    
            starting_date,
            datapoints,
            created_at,
            allocation
        FROM `{BQ_TABLE}`
        WHERE generation_id = '{generation_id}'
    )
    SELECT *
    FROM latest_allocation
    """

    try:
        query_job = client.query(query)
        results = query_job.result()

        rows = list(results)
        if not rows:
            print(f"No results found for generation_id: {generation_id}")
            return None

        row = rows[0]
        print(f"Found data: {row}")

        allocation_array = row.allocation
        if not allocation_array:
            return None

        latest_period = allocation_array[-1]

        portfolio_recommendations = [
            {
                "ticker": alloc["ticker"],
                "allocation_percentage": alloc["allocation_percentage"]
            }
            for alloc in latest_period["allocation_detail"]
        ]

        return {
            'status': 'success',
            'message': 'Optimization completed successfully',
            'portfolio_recommendation': portfolio_recommendations,
            'metadata': {
                'generation_id': row.generation_id,
                'user_id': row.user_id,
                'start_date': row.starting_date,
                'datapoints': row.datapoints,
                'created_at': row.created_at.strftime('%Y-%m-%d %H:%M:%S')
            }
        }

    except Exception as e:
        logger.error(f"Error retrieving results from BigQuery: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving results from BigQuery: {str(e)}"
        )


class OptimizationService:
    def __init__(self):
        self.publisher = pubsub_v1.PublisherClient()
        self.topic_path = self.publisher.topic_path(PROJECT_ID, PUBSUB_TOPIC)

    async def submit_optimization(self, request: OptimizationRequestV2) -> str:
        """
        Submit an optimization request to the Pub/Sub topic.
        """
        try:
            job_id = str(uuid.uuid4())
            volatility = 0.8
            sector_limits = {
                'Financial': 1,
                'Technology': 1,
                'Energy': 1,
                'Healthcare': 1,
                'Industrial': 1,
                'Materials': 1,
                'Other': 1
            }
            if request.volatility:
                volatility = request.volatility
            if request.sector_limits:
                sector_limits = request.sector_limits

            message_data = {
                "job_id": job_id,
                "start_date": request.start_date,
                "granularity": request.granularity,
                "volatility": volatility,
                "sector_limits": sector_limits,
                "user_id": request.user_id
            }
            future = self.publisher.publish(
                self.topic_path,
                data=json.dumps(message_data).encode("utf-8")
            )

            future.result()
            logger.info(f"Optimization request submitted successfully for job_id: {job_id}")
            return job_id
        except Exception as e:
            logger.error(f"Error submitting optimization request: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=str(e)
            )


async def verify_token(x_api_key: str = Header(None, alias="X-API-key")) -> str:
    """
    Verify API key from request header.

    Args:
        x_api_key (str, optional): API key from X-API-key header. Defaults to None.

    Returns:
        str: Validated API key

    Raises:
        HTTPException:
            - 401: If API key is invalid or missing
    """
    if x_api_key != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid Token Bearer key"
        )
    return x_api_key


async def verify_firebase_token(authorization: str = Header(None)) -> dict:
    """
    Verify Firebase authentication token from request header.

    Args:
        authorization (str, optional): Authorization header containing Bearer token.
            Must be in format "Bearer <token>". Defaults to None.

    Returns:
        dict: Decoded Firebase token containing user claims

    Raises:
        HTTPException:
            - 401: If token is invalid, expired, or missing
    """
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


def get_user_portfolio_history(user_id: str) -> Dict[str, Any]:
    """
    Retrieve complete portfolio optimization history for a user from BigQuery.

    Args:
        user_id (str): User identifier (email format)

    Returns:
        Dict[str, Any]: Portfolio history containing:
            - status (str): Operation status ('success' or 'error')
            - message (str): Status description
            - portfolio_history (List[Dict]): List of historical portfolios, each containing:
                - generation_id (str): Optimization generation ID
                - start_date (str): Start date of optimization
                - datapoints (int): Number of data points used
                - created_at (str): Creation timestamp
                - optimized_portfolio_month (str): Month of optimization
                - optimized_portfolio_allocation (List[Dict]): Portfolio allocations excluding benchmark

    Raises:
        HTTPException:
            - 500: If database query fails
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


@app.options("/{rest_of_path:path}")
def preflight_handler():
    headers = {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
        "Access-Control-Allow-Headers": "Authorization, Content-Type, X-API-key",
    }
    return JSONResponse(status_code=200, headers=headers)


def get_optimization_service() -> OptimizationService:
    return OptimizationService()


@app.get("/")
def read_root():
    return {"Status": "OK",
            "Message": "Welcome to Portfolio Optimization API (25-01-15.01)"
            }

@app.post("/optimize/v2", response_model=OptimizationResponse)
async def optimize_v2(
        request: OptimizationRequestV2,
        optimization_service: OptimizationService = Depends(get_optimization_service),
        api_key: str = Depends(verify_token),
        user_claims: dict = Depends(verify_firebase_token)
) -> OptimizationResponse:
    """
       Initialize portfolio optimization with custom parameters.

       Args:
           request (OptimizationRequestV2): Advanced optimization parameters including:
               - start_date (str): Start date for optimization
               - granularity (int): Number of days for optimization window
               - user_id (str): User identifier
               - volatility (float): Target portfolio volatility (0-1)
               - sector_limits (Optional[Dict[str, float]]): Maximum allocation per sector
           optimization_service (OptimizationService): Service for submitting optimization requests
           api_key (str): API key from verify_token dependency
           user_claims (dict): Firebase user claims from verify_firebase_token dependency

       Returns:
           Dict[str, Any]: Response containing:
               - task_id (str): Unique identifier for the optimization task
               - status (str): Initial task status ('processing')
               - message (str): Status description

       Raises:
           HTTPException:
               - 400: If optimization initialization fails
               - 401: If API key validation fails
       """
    try:
        job_id = await optimization_service.submit_optimization(request)
        return OptimizationResponse(
            task_id=job_id,
            status="processing",
            message="Optimization job submitted successfully"
        )

    except Exception as e:
        logger.error(f"Error in submitting optimization: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/optimizers/{task_id}", response_model=OptimizationResult)
def get_optimization_result(
        task_id: str,
        api_key: str = Depends(verify_token),
        user_claims: dict = Depends(verify_firebase_token)
) -> Dict[str, Any]:
    """
    Retrieve optimization results for a specific task.

    Args:
        task_id (str): Unique identifier for the optimization task
        api_key (str): API key from verify_token dependency
        user_claims (dict): Firebase user claims from token verification

    Returns:
        Dict[str, Any]: Optimization results containing:
            - status (str): Task status
            - message (str): Status description
            - portfolio_recommendation (Optional[List[PortfolioAllocation]]):
                Portfolio allocations if available
            - metadata (Optional[Dict[str, Any]]): Additional task information

    Raises:
        HTTPException:
            - 404: If task not found
            - 401: If Firebase authentication fails
    """
    print(f"Received request for task_id: {task_id}")

    result = get_latest_allocation(task_id)
    print(f"Query result: {result}")

    if not result:
        print(f"No result found for task_id: {task_id}")
        raise HTTPException(status_code=404, detail="Task not found")

    return result


@app.get("/users/{user_id}/portfolio-history")
def get_user_portfolios(
        user_id: str,
        api_key: str = Depends(verify_token),
        user_claims: dict = Depends(verify_firebase_token)
) -> Dict[str, Any]:
    """
    Retrieve complete portfolio optimization history for a specific user.

    Args:
        user_id (str): User identifier (email format)
        api_key (str): API key from verify_token dependency
        user_claims (dict): Firebase user claims from token verification
    Returns:
        Dict[str, Any]: Portfolio history containing:
            - status (str): Operation status ('success' or 'error')
            - message (str): Status description
            - portfolio_history (List[Dict]): List of historical portfolios, each containing:
                - generation_id (str): Optimization generation ID
                - start_date (str): Start date of optimization
                - datapoints (int): Number of data points used
                - volatility (float): Target portfolio volatility
                - created_at (str): Creation timestamp
                - optimized_portfolio_month (str): Month of optimization
                - optimized_portfolio_allocation (List[Dict]): Portfolio allocations excluding benchmark
    Raises:
        HTTPException:
            - 400: If database query fails
    """
    try:
        logger.info(f"Fetching portfolio history for user: {user_id}")
        result = get_user_portfolio_history(user_id)
        return result
    except Exception as e:
        logger.error(f"Error fetching user portfolio history: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/health")
def health_check(
        # api_key: str = Depends(verify_token)
) -> Dict[str, str]:
    """
    Check API service health status.

    Returns:
        Dict[str, str]: Health status containing:
            - status (str): Service status ('healthy')
            - timestamp (str): Current timestamp in Asia/Jakarta timezone
    """
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
