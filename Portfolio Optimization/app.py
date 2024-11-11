# -*- coding: utf-8 -*-
from google.cloud import bigquery
import asyncio
from asyncio import Semaphore
from concurrent.futures.thread import ThreadPoolExecutor
from enum import Enum
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Depends, Security, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from granian.constants import Interfaces
from pydantic import BaseModel, field_validator
from typing import Dict, Any, List, Optional
from google.api_core import retry
import uuid
import os
import pytz
import granian

from src.optimizer import run_optimization_pipeline
from src.utils.validators import *

load_dotenv()

# Get API key from environment variable
API_KEY = os.getenv("API_KEY")

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
client = bigquery.Client()


# Pydantic models for request/response
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
    status: str
    message: str
    optimization_id: str
    portfolio_recommendation: List[PortfolioAllocation]
    metadata: Dict[str, Any]


class HealthResponse(BaseModel):
    status: str
    timestamp: str


async def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)) -> str:
    """
    Verify the Bearer token from the request header.

    Args:
        credentials: Bearer token credentials

    Returns:
        str: Token if valid

    Raises:
        HTTPException: If token is invalid
    """
    try:
        token = credentials.credentials
        if token != API_KEY:
            raise HTTPException(
                status_code=401,
                detail="Invalid authentication token"
            )
        return token
    except Exception:
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication credentials"
        )


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class JobResponse(BaseModel):
    job_id: str
    status: JobStatus
    message: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    message: str
    result: Optional[Dict[str, Any]] = None


# In-memory job storage (replace with Redis/database in production)
job_store = {}

# Create a thread pool executor for running CPU-intensive tasks
thread_pool = ThreadPoolExecutor(max_workers=1)

gurobi_semaphore = Semaphore(1)


async def run_optimization_task(
        job_id: str,
        request: OptimizationRequest,
        client: bigquery.Client
) -> None:
    """
    Run portfolio optimization in the background.

    Args:
        job_id (str): Job ID
        request (OptimizationRequest): Optimization request
        client (bigquery.Client): BigQuery client

    Returns:
        None
    """
    try:
        job_store[job_id]["status"] = JobStatus.RUNNING
        job_store[job_id]["message"] = "Optimization in progress"

        optimizer = await asyncio.get_event_loop().run_in_executor(
            thread_pool,
            run_optimization_pipeline,
            request.granularity,
            request.start_date
        )
        results_df = optimizer.final_result
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

        # Create base record
        allocation_array = []
        for period, group in results_df.groupby('period'):
            allocation_details = [
                {
                    "ticker": row['ticker'],
                    "allocation_percentage": round(row['allocations'] * 100, 2),
                    "allocation_idr": row['allocations_idr'],
                    "pnl_percentage": row['pnl_percentage'],
                    "pnl_idr": row['pnl_idr'],
                    "end_of_period_value": row['end_of_period_value'],
                }
                for _, row in group.iterrows() if row['ticker'] != 'LQ45'
            ]

            period_struct = {
                "period": period,
                "allocation_detail": allocation_details
            }
            allocation_array.append(period_struct)

        wib = pytz.timezone('Asia/Jakarta')
        current_timestamp = datetime.now(wib)
        data = [{
            'generation_id': job_id,
            'user_id': request.user_id,
            'starting_date': request.start_date,
            'datapoints': request.granularity,
            'created_at': current_timestamp.isoformat(),
            'allocation': allocation_array
        }]

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
        table_id = 'km-data-dev.tuan_takur.user_generated_optimization'
        job_config = bigquery.LoadJobConfig(
            schema=schema,
            write_disposition="WRITE_APPEND"
        )

        @retry.Retry(
            initial=1.0,  # Initial delay in seconds
            maximum=60.0,  # Maximum delay
            multiplier=2.0,  # Multiply delay by this factor after each failure
            predicate=retry.if_exception_type(Exception),
            timeout=600.0  # Total timeout in seconds
        )
        def execute_bigquery_job():
            job = client.load_table_from_json(data, table_id, job_config=job_config)
            return job.result()

        # Execute with retry
        await asyncio.get_event_loop().run_in_executor(
            thread_pool,
            execute_bigquery_job
        )

        # Update job store with results
        result = {
            'status': 'success',
            'message': 'Optimization completed and results stored',
            'optimization_id': job_id,
            'portfolio_recommendation': portfolio_recommendations,
            'metadata': {
                'generation_id': job_id,
                'user_id': request.user_id,
                'start_date': request.start_date,
                'datapoints': request.granularity,
                'created_at': current_timestamp.strftime('%Y-%m-%d %H:%M:%S')
            }
        }

        job_store[job_id].update({
            "status": JobStatus.COMPLETED,
            "message": "Optimization completed successfully",
            "result": result
        })
    except Exception as e:
        job_store[job_id].update({
            "status": JobStatus.FAILED,
            "message": f"optimization failed: {str(e)}"
        })


@app.get("/")
async def read_root():
    return {"message": "Welcome to the Portfolio Optimization API"}


@app.post("/optimize", response_model=JobResponse)
async def optimize(
        request: OptimizationRequest,
        background_tasks: BackgroundTasks,
        authenticated: bool = Depends(verify_token)
) -> Dict[str, Any]:
    """
    Start portfolio optimization in background

    Args:
        request (OptimizationRequest): Request containing start_date and granularity
        authenticated (bool): Flag indicating if user is authenticated
        background_tasks (BackgroundTasks): Background task manager

    Returns:
        Dict[str, Any]: Job ID, status, and message

    Raises:
        HTTPException: If optimization fails
    """
    if not authenticated:
        raise HTTPException(status_code=401, detail="Authentication required")

    try:
        job_id = str(uuid.uuid4())
        job_store[job_id] = {
            "status": JobStatus.PENDING,
            "message": "Optimization is queued"
        }

        background_tasks.add_task(
            run_optimization_task,
            job_id,
            request,
            client
        )
        return {
            "job_id": job_id,
            "status": JobStatus.PENDING,
            "message": "Optimization started"
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/optimize/status/{job_id}", response_model=JobStatusResponse)
async def get_optimization_status(
        job_id: str,
        authenticated: bool = Depends(verify_token)
) -> Dict[str, Any]:
    """
    Get optimization status by job ID

    Args:
        job_id (str): Job ID
        authenticated (bool): Flag indicating if user is authenticated

    Returns:
        Dict[str, Any]: Job status and message

    Raises:
        HTTPException: If job ID is not found
    """
    if not authenticated:
        raise HTTPException(status_code=401, detail="Authentication required")

    if job_id not in job_store:
        raise HTTPException(status_code=404, detail="Job ID not found")

    job_info = job_store[job_id]
    if not job_info:
        raise HTTPException(
            status_code=404,
            detail=f"Job {job_id} not found"
        )

    return {
        "job_id": job_id,
        **job_info
    }


@app.get("/health", response_model=HealthResponse)
async def health_check(authenticated: bool = Depends(verify_token)) -> Dict[str, str]:
    """Health check endpoint"""
    if not authenticated:
        raise HTTPException(status_code=401, detail="Authentication required")
    return {
        'status': 'healthy',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }


if __name__ == "__main__":
    granian.Granian(
        target=app,
        address="0.0.0.0",
        port=8010,
        interface=Interfaces.ASGI,
    )