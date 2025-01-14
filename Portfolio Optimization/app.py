import requests
from google.cloud import bigquery
from dotenv import load_dotenv
from google.oauth2 import service_account
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer
from fastapi.responses import JSONResponse
from granian.constants import Interfaces
from pydantic import BaseModel, field_validator
from typing import Dict, Any, List, Optional
from google.api_core import retry
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
    version="1.0.2",
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


def run_optimization_task_v2(
        task_id: str,
        request: Union[OptimizationRequest, OptimizationRequestV2],
) -> None:
    """
    Execute portfolio optimization task and store results in BigQuery.

    Args:
        task_id (str): Unique identifier for the optimization task
        request (Union[OptimizationRequest, OptimizationRequestV2]): Optimization parameters
            - OptimizationRequest: Basic request with default parameters
            - OptimizationRequestV2: Advanced request with custom volatility and sector limits

    Raises:
        HTTPException:
            - 500: If optimization fails or database operations fail

    Notes:
        - For OptimizationRequest, uses default volatility (0.5) and sector limits
        - For OptimizationRequestV2, uses custom volatility and sector limits if provided
        - Results are stored in both memory (optimization_results) and BigQuery
        - Sends notification upon successful completion
    """
    try:
        logger.info(f"Running optimization for task_id: {task_id}")
        volatility = 0.5  # 20% default volatility
        sector_limits = {
            'Financial': 0.5,
            'Technology': 0.5,
            'Energy': 0.5,
            'Healthcare': 0.5,
            'Industrial': 0.5,
            'Materials': 0.5,
            'Other': 0.5
        }
        if isinstance(request, OptimizationRequestV2):
            volatility = request.volatility
            if request.sector_limits:
                sector_limits = request.sector_limits

        optimizer = run_optimization_pipeline(
            request.granularity,
            volatility,
            request.start_date,
            sector_limits

        )

        results_df = optimizer.final_result
        logger.info(f"Optimization completed successfully for task_id: {task_id}")
        current_period = results_df['period'].max()
        current_allocations = results_df[
            (results_df['period'] == current_period) &
            (results_df['ticker'] != 'LQ45')
            ]

        portfolio_recommendations = [
            {
                "ticker": row['ticker'],
                "allocation_percentage": round(row['allocations'] * 100, 2)
            }
            for _, row in current_allocations.iterrows()
        ]

        allocation_array = []
        for period, group in results_df.groupby('period'):
            allocation_details = [
                {
                    "ticker": row['ticker'],
                    "industry": row['industry'],
                    "allocation_percentage": round(row['allocations'] * 100, 2),
                    "allocation_idr": row['allocations_idr'],
                    "pnl_percentage": row['pnl_percentage'],
                    "pnl_idr": row['pnl_idr'],
                    "end_of_period_value": row['end_of_period_value'],
                }
                for _, row in group.iterrows()
            ]

            allocation_array.append({
                "period": period,
                "allocation_detail": allocation_details
            })

        wib = pytz.timezone('Asia/Jakarta')
        current_timestamp = datetime.now(wib)

        optimization_results[task_id] = {
            'status': 'success',
            'message': 'Optimization completed successfully',
            'portfolio_recommendation': portfolio_recommendations,
            'metadata': {
                'generation_id': task_id,
                'user_id': request.user_id,
                'start_date': request.start_date,
                'datapoints': request.granularity,
                'volatility': request.volatility,
                'created_at': current_timestamp.isoformat()
            }
        }

        data = [{
            'generation_id': task_id,
            'user_id': request.user_id,
            'starting_date': request.start_date,
            'datapoints': request.granularity,
            'volatility': request.volatility,
            'created_at': current_timestamp.isoformat(),
            'allocation': allocation_array
        }]

        logger.info(f"Saving results to BigQuery for task_id: {task_id}")
        schema = [
            bigquery.SchemaField("generation_id", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("user_id", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("starting_date", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("datapoints", "INTEGER", mode="REQUIRED"),
            bigquery.SchemaField("volatility", "FLOAT", mode="REQUIRED"),
            bigquery.SchemaField("created_at", "TIMESTAMP", mode="REQUIRED"),
            bigquery.SchemaField(
                "allocation",
                "RECORD",
                mode="REPEATED",
                fields=[
                    bigquery.SchemaField("period", "STRING", mode="REQUIRED"),
                    bigquery.SchemaField(
                        "allocation_detail",
                        "RECORD",
                        mode="REPEATED",
                        fields=[
                            bigquery.SchemaField("ticker", "STRING", mode="REQUIRED"),
                            bigquery.SchemaField("industry", "STRING", mode="REQUIRED"),
                            bigquery.SchemaField("allocation_percentage", "FLOAT", mode="REQUIRED"),
                            bigquery.SchemaField("allocation_idr", "FLOAT", mode="REQUIRED"),
                            bigquery.SchemaField("pnl_percentage", "FLOAT", mode="REQUIRED"),
                            bigquery.SchemaField("pnl_idr", "FLOAT", mode="REQUIRED"),
                            bigquery.SchemaField("end_of_period_value", "FLOAT", mode="REQUIRED")
                        ]
                    )
                ]
            )
        ]

        try:
            table_id = f'{BQ_TABLE}'
            logger.info(f"table_id has been set {table_id}")
            job_config = bigquery.LoadJobConfig(
                schema=schema,
                # create_disposition="CREATE_NEVER",
                write_disposition="WRITE_APPEND"
            )
            logger.info(f"job_config has been set")

            @retry.Retry(
                initial=1.0,
                maximum=60.0,
                multiplier=2.0,
                predicate=retry.if_exception_type(Exception),
                timeout=600.0
            )
            def execute_bigquery_job():
                job = client.load_table_from_json(data, table_id, job_config=job_config)
                return job.result()

            logger.info(f"executing_bigquery_job")
            execute_bigquery_job()
            logger.info(f"Results have been appended to BigQuery table for task_id: {task_id}")
            logger.info(f"Optimization completed successfully for task_id: {task_id}")
            notify_url = os.getenv("NOTIFY_URL")
            payload = {
                "generation_id": task_id
            }
            response = requests.post(notify_url, json=payload)
            if response.status_code == 200:
                logger.info(f"Notification sent successfully for task_id: {task_id}")
            else:
                logger.error(f"Error sending notification for task_id: {task_id}")


        except Exception as e:
            logger.error(f"Error writing to BigQuery: {str(e)}", exc_info=True)
            optimization_results[task_id] = {
                'status': 'error',
                'message': f"Error writing results to BigQuery: {str(e)}"
            }
            raise HTTPException(
                status_code=500,
                detail=str(e)
            )

    except Exception as e:
        logger.error(f"Error during optimization: {str(e)}", exc_info=True)
        optimization_results[task_id] = {
            'status': 'error',
            'message': str(e)
        }
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


@app.get("/")
def read_root():
    return {"Status": "OK",
            "Message": "Welcome to Portfolio Optimization API (24-12-23.01)"
            }


@app.post("/optimize", response_model=OptimizationResponse)
def optimize(
        request: OptimizationRequest,
        background_tasks: BackgroundTasks,
        api_key: str = Depends(verify_token),
        # user_claims: dict = Depends(verify_firebase_token)
) -> Dict[str, Any]:
    """
    Initialize portfolio optimization with default parameters.

    Args:
        request (OptimizationRequest): Basic optimization parameters including:
            - start_date (str): Start date for optimization
            - granularity (int): Number of days for optimization window
            - user_id (str): User identifier
        background_tasks (BackgroundTasks): FastAPI background tasks handler
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
        task_id = str(uuid.uuid4())
        optimization_results[task_id] = {
            'status': 'processing',
            'message': 'Optimization in progress'
        }
        logger.info(
            f"Optimization with default params started for task_id: {task_id}, user_id: {request.user_id}, start_date: {request.start_date}, granularity: {request.granularity}")
        background_tasks.add_task(
            run_optimization_task_v2,
            task_id,
            request
        )
        return {
            "task_id": task_id,
            "status": "processing",
            "message": "Optimization started"
        }

    except Exception as e:
        logger.error(f"Error starting optimization: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/optimize/v2", response_model=OptimizationResponse)
def optimize_v2(
        request: OptimizationRequestV2,
        background_tasks: BackgroundTasks,
        api_key: str = Depends(verify_token),
        # user_claims: dict = Depends(verify_firebase_token)
) -> Dict[str, Any]:
    """
       Initialize portfolio optimization with custom parameters.

       Args:
           request (OptimizationRequestV2): Advanced optimization parameters including:
               - start_date (str): Start date for optimization
               - granularity (int): Number of days for optimization window
               - user_id (str): User identifier
               - volatility (float): Target portfolio volatility (0-1)
               - sector_limits (Optional[Dict[str, float]]): Maximum allocation per sector
           background_tasks (BackgroundTasks): FastAPI background tasks handler
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
        task_id = str(uuid.uuid4())
        optimization_results[task_id] = {
            'status': 'processing',
            'message': 'Optimization in progress'
        }
        logger.info(
            f"Optimization with custom params started for task_id: {task_id}, user_id: {request.user_id}, start_date: {request.start_date}, granularity: {request.granularity}, volatility: {request.volatility}, sector_limits: {request.sector_limits}")
        background_tasks.add_task(
            run_optimization_task_v2,
            task_id,
            request
        )
        return {
            "task_id": task_id,
            "status": "processing",
            "message": "Optimization started"
        }

    except Exception as e:
        logger.error(f"Error starting optimization: {str(e)}")
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
