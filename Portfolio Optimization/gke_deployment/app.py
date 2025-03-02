import os
import json
import logging
from typing import Optional, Dict, Any, List, Union
from google.cloud import pubsub_v1, bigquery
from datetime import datetime
from dotenv import load_dotenv
import pytz
import pandas as pd
from src.optimizer import run_optimization_pipeline
from fastapi import FastAPI
import uvicorn
from threading import Thread
import signal
from google.cloud.pubsub_v1.subscriber.message import Message
from google.api_core import retry
from concurrent.futures import Future

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI()
fastapi_thread = None


@app.get('/health')
async def health() -> Dict[str, str]:
    """
    Health check endpoint for the FastAPI application.

    Returns:
        Dict[str, str]: A dictionary containing the health status
    """
    return {'status': 'healthy'}


class OptimizerService:
    """
    Service for handling portfolio optimization tasks using Google Cloud services.
    This service manages the connection to Google Cloud Pub/Sub for receiving optimization
    requests and BigQuery for storing optimization results.
    """

    def __init__(self):
        """
        Initialize the OptimizerService with required Google Cloud configurations.

        Raises:
            ValueError: If any required environment variables are missing
        """
        required_env = ['GOOGLE_CLOUD_PROJECT', 'PUBSUB_SUBSCRIPTION', 'BQ_TABLE']
        missing_env = [env for env in required_env if not os.getenv(env)]
        if missing_env:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_env)}")
        self.project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
        self.subscription_name = os.getenv('PUBSUB_SUBSCRIPTION')
        self.bq_table = os.getenv('BQ_TABLE')

        self.subscriber = pubsub_v1.SubscriberClient()
        self.subscription_path = self.subscriber.subscription_path(
            self.project_id,
            self.subscription_name
        )
        self.bigquery_client = bigquery.Client()

    def save_to_bigquery(
            self,
            job_id: str,
            user_id: str,
            granularity: int,
            volatility: float,
            start_date: str,
            sector_limits: Optional[Dict[str, float]],
            results: pd.DataFrame
    ) -> None:
        """
        Save optimization results to BigQuery with comprehensive schema and error handling.

        Args:
            job_id (str): Unique identifier for the optimization job
            user_id (str): Identifier for the user who requested the optimization
            granularity (int): Time period granularity for the optimization
            volatility (float): Target volatility for the portfolio
            start_date (str): Start date for the optimization period
            sector_limits (Optional[Dict[str, float]]): Sector allocation limits
            results (pd.DataFrame): DataFrame containing optimization results

        Raises:
            Exception: If there's an error saving data to BigQuery
        """

        try:
            allocation_array = []
            for period, group in results.groupby('period'):
                allocation_details = [
                    {
                        "ticker": row['ticker'],
                        "industry": row['industry'],
                        "allocation_percentage": float(round(row['allocations'] * 100, 2)),
                        "allocation_idr": float(row['allocations_idr']),
                        "pnl_percentage": float(row['pnl_percentage']),
                        "pnl_idr": float(row['pnl_idr']),
                        "end_of_period_value": float(row['end_of_period_value'])
                    }
                    for _, row in group.iterrows()
                ]
                allocation_array.append({
                    "period": period,
                    "allocation_detail": allocation_details
                })

            wib = pytz.timezone('Asia/Jakarta')
            timestamp = datetime.now(wib).isoformat()

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

            data = [{
                'generation_id': job_id,
                'user_id': user_id,
                'starting_date': start_date,
                'datapoints': granularity,
                'volatility': volatility,
                'created_at': timestamp,
                'allocation': allocation_array
            }]

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
                job = self.bigquery_client.load_table_from_json(
                    data,
                    self.bq_table,
                    job_config=job_config
                )
                return job.result()

            execute_bigquery_job()
            logger.info(f"Successfully saved results for job {job_id}")

        except Exception as e:
            logger.error(f"Error saving to BigQuery: {str(e)}", exc_info=True)
            raise

    def process_message(self, message: Message) -> None:
        """
        Process a single message from Pub/Sub containing optimization parameters.

        Args:
            message (Message): The Pub/Sub message containing optimization parameters

        Note:
            The message is acknowledged even if processing fails to prevent
            infinite retries of failed messages.
        """
        try:
            data = json.loads(message.data.decode('utf-8'))
            start_date = str(data['start_date'])
            volatility = float(data['volatility'])
            logger.info(f"Processing optimization job: {data['job_id']}")

            optimizer = run_optimization_pipeline(
                granularity=data['granularity'],
                volatility=data['volatility'],
                start_date=data['start_date'],
                sector_limits=data.get('sector_limits')
            )

            self.save_to_bigquery(
                data['job_id'],
                data['user_id'],
                data['granularity'],
                data['volatility'],
                data['start_date'],
                data.get('sector_limits'),
                optimizer.final_result
            )

            message.ack()
            logger.info(f"Successfully processed job {data['job_id']}")

        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            logger.error(f"Message data: {message.data.decode('utf-8')}")
            message.ack()

    def run(self) -> None:
        """
        Run the optimization service with a FastAPI health check endpoint.

        This method:
        1. Starts a FastAPI server in a separate thread for health checks
        2. Subscribes to the Pub/Sub subscription for optimization requests
        3. Sets up signal handlers for graceful shutdown
        4. Maintains the subscription until interrupted

        Note:
            The service runs indefinitely until interrupted by a SIGTERM or SIGINT signal.
        """
        logger.info(f"Starting service on {self.subscription_path}")

        global fastapi_thread
        fastapi_thread = Thread(
            target=lambda: uvicorn.run(
                app,
                host="0.0.0.0",
                port=8080,
                log_level="info"
            )
        )
        fastapi_thread.daemon = True
        fastapi_thread.start()

        streaming_pull_future = self.subscriber.subscribe(
            self.subscription_path,
            callback=self.process_message
        )

        def cleanup(signum, frame):
            logger.info("Received shutdown signal, cleaning up...")
            streaming_pull_future.cancel()
            self.subscriber.close()
            logger.info("Shutting down health check endpoint...")

        signal.signal(signal.SIGTERM, cleanup)
        signal.signal(signal.SIGINT, cleanup)

        try:
            streaming_pull_future.result()
        except Exception as e:
            logger.error(f"Streaming pull future terminated: {str(e)}")
            streaming_pull_future.cancel()
        finally:
            self.subscriber.close()


if __name__ == "__main__":
    service = OptimizerService()
    service.run()
