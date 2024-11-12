from google.cloud import bigquery
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer
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
            print(f"No results found for generation_id: {generation_id}")  # For debugging
            return None

        row = rows[0]  # Get the first row

        # Debug print the row data
        print(f"Found data: {row}")  # For debugging

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
        print(f"Error in get_latest_allocation: {str(e)}")  # For debugging
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
        # Run optimization
        optimizer = run_optimization_pipeline(
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

        # Prepare data for BigQuery
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
                'created_at': current_timestamp.strftime('%Y-%m-%d %H:%M:%S')
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
            initial=1.0,
            maximum=60.0,
            multiplier=2.0,
            predicate=retry.if_exception_type(Exception),
            timeout=600.0
        )
        def execute_bigquery_job():
            job = client.load_table_from_json(data, table_id, job_config=job_config)
            return job.result()

        execute_bigquery_job()

    except Exception as e:
        optimization_results[task_id] = {
            'status': 'error',
            'message': str(e)
        }


async def verify_token(x_api_key: str = Header(None, alias="X-API-key")) -> str:
    """Verify the Bearer token from the request header."""
    if x_api_key != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid Token Bearer key"
        )
    return x_api_key

@app.get("/")
def read_root():
    return {"Status": "OK",
            "Message": "Welcome to Portfolio Optimization API"
    }

@app.post("/optimize", response_model=OptimizationResponse)
def optimize(
        request: OptimizationRequest,
        background_tasks: BackgroundTasks,
        api_key: str = Depends(verify_token)
) -> Dict[str, Any]:
    """Start portfolio optimization"""
    try:
        task_id = str(uuid.uuid4())
        optimization_results[task_id] = {
            'status': 'processing',
            'message': 'Optimization in progress'
        }

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
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/optimizers/{task_id}", response_model=OptimizationResult)
def get_optimization_result(
        task_id: str,
        api_key: str = Depends(verify_token)
) -> Dict[str, Any]:
    """Get optimization results from BigQuery"""
    print(f"Received request for task_id: {task_id}")  # Debug log

    result = get_latest_allocation(task_id)
    print(f"Query result: {result}")  # Debug log

    if not result:
        print(f"No result found for task_id: {task_id}")  # Debug log
        raise HTTPException(status_code=404, detail="Task not found")

    return result


@app.get("/health")
def health_check(api_key: str = Depends(verify_token)) -> Dict[str, str]:
    """Health check endpoint"""
    return {
        'status': 'healthy',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }


if __name__ == "__main__":
    granian.Granian(
        target="app:app",
        address="0.0.0.0",
        port=8080,
        interface=Interfaces.ASGI,
        workers=2,
    ).serve()
