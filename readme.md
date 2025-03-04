# Portfolio Optimization Using Gurobi

## Description

This project implements an API for portfolio optimization and analysis. It leverages various finance data from Yahoo and optimization techniques to provide users with portfolio recommendations based on their specified parameters, such as start date, granularity, volatility, and sector limits. The API integrates with Google Cloud services like Pub/Sub, BigQuery, and Google Kubernetes Engine for asynchronous task processing and data storage.

## Technologies Used

-   Python
-   FastAPI
-   Gurobi
-   Google Cloud Pub/Sub
-   Google Cloud BigQuery
-   yfinance
-   dotenv
-   pandas
-   uvicorn
-   granian


## Flow Explanation

1.  **API Request**: The user sends a request to the API with optimization parameters (start date, granularity, volatility, sector limits, user ID).
2.  **Request Validation**: The API validates the request parameters.
3.  **Task Submission**: The API submits an optimization task to a Google Cloud Pub/Sub topic.
4.  **Asynchronous Processing**: A worker service (OptimizerService) subscribes to the Pub/Sub topic and receives the optimization task.
5.  **Data Loading**: The worker service loads stock data, industry data, and benchmark data from Google Cloud BigQuery and yfinance.
6.  **Data Preparation**: The data is prepared for the optimization process, including calculating covariance matrices and expected returns.
7.  **Optimization**: The Gurobi optimizer is used to determine the optimal portfolio allocation based on the given parameters and constraints.
8.  **Result Storage**: The optimization results are stored in Google Cloud BigQuery.
9.  **Result Retrieval**: The user can query the API to retrieve the optimization results by task ID.
10. **Portfolio History**: The user can also retrieve their complete portfolio optimization history.

## Result

The API returns a portfolio recommendation containing:

-   **Task ID**: A unique identifier for the optimization task.
-   **Status**: The status of the optimization task (e.g., "processing", "success", "error").
-   **Message**: A descriptive message about the task status.
-   **Portfolio Allocation**: A list of recommended stock allocations, including the ticker symbol and allocation percentage.
-   **Metadata**: Additional information about the optimization, such as the start date, data points, and creation timestamp.
