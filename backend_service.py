import time
import os
import requests
from google.cloud import storage


class JobManager:
    def __init__(self):
        self.jobs = {}

    def create_job(self, user_email):
        job_id = f"{user_email.replace('@', '_').replace('.', '_')}_job_{int(time.time())}"
        self.jobs[job_id] = {"status": "created", "user_email": user_email}
        return job_id

    def update_job_status(self, job_id, status):
        if job_id in self.jobs:
            self.jobs[job_id]["status"] = status
        else:
            raise ValueError(f"Job ID {job_id} not found!")

    def get_job_status(self, job_id):
        return self.jobs.get(job_id, {}).get("status", "unknown")


class GCSManager:
    def __init__(self, bucket_name):
        self.storage_client = storage.Client()
        self.bucket = self.storage_client.bucket(bucket_name)

    def upload_file(self, local_path, gcs_path):
        """Upload a file to a GCS bucket."""
        blob = self.bucket.blob(gcs_path)
        blob.upload_from_filename(local_path)
        return f"gs://{self.bucket.name}/{gcs_path}"

    def list_files(self, prefix):
        """List files in a GCS bucket with a specific prefix."""
        return [blob.name for blob in self.bucket.list_blobs(prefix=prefix)]


class OptimizationAPI:
    def __init__(self, api_url, api_key):
        self.api_url = api_url
        self.api_key = api_key

    def start_optimization(self, user_email, budget, job_id):
        payload = {
            "start_date": "2024-01-01",  # Example payload
            "granularity": 30,
            "user_id": user_email
        }
        headers = {"X-API-key": self.api_key}
        response = requests.post(f"{self.api_url}/optimize", json=payload, headers=headers)
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Optimization failed: {response.text}")


class BackendService:
    def __init__(self, gcs_bucket_name, optimization_url, optimization_api_key):
        self.job_manager = JobManager()
        self.gcs_manager = GCSManager(gcs_bucket_name)
        self.optimization_api = OptimizationAPI(optimization_url, optimization_api_key)

    def process_request(self, user_email, budget):
        # Step 1: Create a new job
        job_id = self.job_manager.create_job(user_email)
        print(f"Job {job_id} created for user {user_email}")

        # Step 2: Call optimization API
        try:
            optimization_result = self.optimization_api.start_optimization(user_email, budget, job_id)
            print(f"Optimization result: {optimization_result}")

            # Step 3: Update job status
            self.job_manager.update_job_status(job_id, "completed")
            print(f"Job {job_id} completed.")
            return {
                "job_id": job_id,
                "status": "completed",
                "result": optimization_result
            }

        except Exception as e:
            print(f"Error during optimization: {e}")
            self.job_manager.update_job_status(job_id, "failed")
            raise