from google.cloud import bigquery
from dotenv import load_dotenv
from google.oauth2 import service_account
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer
from granian.constants import Interfaces
from pydantic import BaseModel, field_validator
from typing import Dict, Any, List, Optional
from google.api_core import retry
from firebase_admin import auth, credentials, initialize_app
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

# Get API key from environment variable
API_KEY = os.getenv("API_KEY")
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
    # Firebase already initialized
    pass

# Security scheme
security = HTTPBearer()

# Initialize FastAPI app
app = FastAPI(
    title="Portfolio Optimization API",
    description="API for portfolio optimization and analysis",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize BigQuery client
# Store for optimization results
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
    Get the latest allocation for a specific generation_id from BigQuery
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
        FROM `km-data-dev.tuan_takur.user_generated_optimization`
        WHERE generation_id = '{generation_id}'
    )
    SELECT *
    FROM latest_allocation
    """

    try:
        # Remove the job_config since we're using f-string
        query_job = client.query(query)
        results = query_job.result()

        # Convert results to list to check if empty
        rows = list(results)
        if not rows:
            print(f"No results found for generation_id: {generation_id}")
            return None

        row = rows[0]

        # Debug print the row data
        print(f"Found data: {row}")

        # Get the last period's allocation from the allocation array
        allocation_array = row.allocation
        if not allocation_array:
            return None

        latest_period = allocation_array[-1]

        # Format the portfolio recommendations
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


def run_optimization_task(
        task_id: str,
        request: OptimizationRequest,
) -> None:
    """
    Run the optimization task and store results
    """
    try:
        logger.info(f"Running optimization for task_id: {task_id}")
        optimizer = run_optimization_pipeline(
            request.granularity,
            request.start_date
        )

        results_df = optimizer.final_result
        logger.info(f"Optimization completed successfully for task_id: {task_id}")
        current_period = results_df['period'].max()
        current_allocations = results_df[
            (results_df['period'] == current_period) &
            (results_df['ticker'] != 'LQ45')
            ]

        # Prepare portfolio recommendations
        portfolio_recommendations = [
            {
                "ticker": row['ticker'],
                "allocation_percentage": round(row['allocations'] * 100, 2)
            }
            for _, row in current_allocations.iterrows()
        ]

        # Prepare data for BigQuery
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

        # Store results
        optimization_results[task_id] = {
            'status': 'success',
            'message': 'Optimization completed successfully',
            'portfolio_recommendation': portfolio_recommendations,
            'metadata': {
                'generation_id': task_id,
                'user_id': request.user_id,
                'start_date': request.start_date,
                'datapoints': request.granularity,
                'created_at': current_timestamp.isoformat()
            }
        }

        # Prepare data for BigQuery
        data = [{
            'generation_id': task_id,
            'user_id': request.user_id,
            'starting_date': request.start_date,
            'datapoints': request.granularity,
            'created_at': current_timestamp.isoformat(),
            'allocation': allocation_array
        }]

        # Save to BigQuery
        logger.info(f"Saving results to BigQuery for task_id: {task_id}")
        schema = [
            bigquery.SchemaField("generation_id", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("user_id", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("starting_date", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("datapoints", "INTEGER", mode="REQUIRED"),
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
            table_id = 'km-data-dev.tuan_takur.user_generated_optimization'
            logger.info(f"table_id has been set {table_id}")
            job_config = bigquery.LoadJobConfig(
                schema=schema,
                create_disposition="CREATE_NEVER",
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
        decoded_token = auth.verify_id_token(token)
        return decoded_token  # You can return claims like uid for further use
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {str(e)}")


def get_user_portfolio_history(user_id: str) -> Dict[str, Any]:
    """
    Get user's portfolio history from BigQuery
    """
    query = f"""
    SELECT *
    FROM `km-data-dev.tuan_takur.user_generated_optimization`
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
            "Message": "Welcome to Portfolio Optimization API (24-11-26.01)"
            }


@app.post("/optimize", response_model=OptimizationResponse)
def optimize(
        request: OptimizationRequest,
        background_tasks: BackgroundTasks,
        api_key: str = Depends(verify_token),
        user_claims: dict = Depends(verify_firebase_token)
)-> Dict[str, Any]:
    """Start portfolio optimization"""
    try:
        task_id = str(uuid.uuid4())
        # user_id = user_claims['uid']
        optimization_results[task_id] = {
            'status': 'processing',
            'message': 'Optimization in progress'
        }
        logger.info(
            f"Optimization started for task_id: {task_id}, user_id: {request.user_id}, start_date: {request.start_date}, granularity: {request.granularity}")
        background_tasks.add_task(
            run_optimization_task,
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
        api_key: str = Depends(verify_token)
) -> Dict[str, Any]:
    """Get optimization results from BigQuery"""
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
        api_key: str = Depends(verify_token)
) -> Dict[str, Any]:
    """Get all portfolio optimization history for a specific user"""
    try:
        logger.info(f"Fetching portfolio history for user: {user_id}")
        result = get_user_portfolio_history(user_id)
        return result
    except Exception as e:
        logger.error(f"Error fetching user portfolio history: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/health")
def health_check(api_key: str = Depends(verify_token)) -> Dict[str, str]:
    """Health check endpoint"""
    WIB = pytz.timezone('Asia/Jakarta')
    return {
        'status': 'healthy',
        'timestamp': datetime.now(WIB).strftime('%Y-%m-%d %H:%M:%S')
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
