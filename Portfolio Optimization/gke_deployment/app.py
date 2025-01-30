import os
import json
import logging
from google.cloud import pubsub_v1, bigquery
from datetime import datetime
from dotenv import load_dotenv
import pytz
from src.optimizer import run_optimization_pipeline
from fastapi import FastAPI
import uvicorn
from threading import Thread
import signal

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI()
fastapi_thread = None


@app.get('/health')
async def health():
    return {'status': 'healthy'}


class OptimizerService:
    def __init__(self):
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

    def save_to_bigquery(self, job_id, user_id, granularity, volatility,
                         start_date, sector_limits, results):
        """Save optimization results to BigQuery"""
        try:
            allocation_array = []
            for period, group in results.groupby('period'):
                allocation_details = [
                    {
                        "ticker": row['ticker'],
                        "allocation_percentage": float(round(row['allocations'] * 100, 2)),
                        "allocation_idr": float(row['allocations_idr'])
                    }
                    for _, row in group.iterrows()
                ]
                allocation_array.append({
                    "period": period,
                    "allocation_detail": allocation_details
                })

            wib = pytz.timezone('Asia/Jakarta')
            timestamp = datetime.now(wib)
            row = {
                'generation_id': job_id,
                'user_id': user_id,
                'starting_date': start_date,
                'granularity': granularity,
                'volatility': volatility,
                'sector_limits': sector_limits,
                'created_at': timestamp.isoformat(),
                'allocation': allocation_array
            }

            table = self.bigquery_client.get_table(self.bq_table)
            errors = self.bigquery_client.insert_rows_json(table, [row])

            if errors:
                raise Exception(f"Error inserting rows: {errors}")

            logger.info(f"Successfully saved results for job {job_id}")

        except Exception as e:
            logger.error(f"Error saving to BigQuery: {str(e)}")
            raise

    def process_message(self, message):
        """Process a single message from Pub/Sub"""
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

    def run(self):
        """Run the service"""
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
