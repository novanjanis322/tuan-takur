import logging
import argparse
from src.optimizer import run_optimization_pipeline
from typing import Tuple, Optional
from src.settings import PROCESSED_DATA_DIR
from src.utils.validators import validate_start_date, validate_granularity

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_user_inputs() -> Tuple[str, int]:
    """
    Get and validate user inputs for optimization parameters.

    This function handles both command-line arguments and interactive input for
    start date and granularity parameters. It validates all inputs before returning.

    Returns:
        Tuple[str, int]: A tuple containing:
            - start_date (str): Validated start date in 'YYYY-MM-DD' format
            - granularity (int): Validated granularity value between 1 and 252

    Raises:
        ValueError: If the provided inputs are invalid after all retry attempts
        SystemExit: If the user interrupts the input process (Ctrl+C)
    """
    parser = argparse.ArgumentParser(description='Portfolio Optimization Parameters')

    # Add command-line arguments
    parser.add_argument(
        '--start-date',
        type=str,
        help='Start date in YYYY-MM-DD format'
    )
    parser.add_argument(
        '--granularity',
        type=int,
        help='Granularity in days'
    )

    args = parser.parse_args()

    # Get and validate start date
    if not args.start_date:
        while True:
            try:
                start_date = input("Enter start date (YYYY-MM-DD): ")
                start_date = validate_start_date(start_date)
                break
            except ValueError as e:
                print(f"Error: {str(e)}")
                print("Please try again.")
    else:
        start_date = validate_start_date(args.start_date)

    # Get and validate granularity
    if args.granularity is None:
        while True:
            try:
                granularity_input = input("Enter granularity (days, default 30): ").strip()
                if granularity_input == "":
                    granularity = 30
                    break
                granularity = validate_granularity(granularity_input)
                break
            except ValueError as e:
                print(f"Error: {str(e)}")
                print("Please try again.")
    else:
        try:
            granularity = validate_granularity(args.granularity)
        except ValueError as e:
            logger.warning(f"Invalid granularity provided: {str(e)}")
            logger.info("Using default granularity of 30 days")
            granularity = 30

    return start_date, granularity


def main() -> None:
    """
    Main function to run the portfolio optimization pipeline.

    This function:
    1. Gets and validates user inputs
    2. Runs the optimization pipeline
    3. Saves results to CSV
    4. Displays summary statistics

    The results are saved to the processed data directory with the filename
    format: 'optimization_results_from_{start_date}.csv'

    Raises:
        Exception: Any error that occurs during the optimization process will be
                  logged and re-raised
        FileNotFoundError: If the output directory doesn't exist or isn't writable
    """

    try:
        # Get validated inputs
        start_date, granularity = get_user_inputs()

        logger.info(f"Starting optimization from {start_date} "
                    f"with granularity {granularity}")

        # Run optimization pipeline
        optimizer = run_optimization_pipeline(
            granularity=granularity,
            start_date=start_date
        )

        # Save results
        results_df = optimizer.final_result
        output_path = PROCESSED_DATA_DIR / f"optimization_results_from_{start_date}_with_{granularity}_datapoints.csv"
        results_df.to_csv(output_path, index=False)

        logger.info(f"Optimization completed successfully. Results saved to {output_path}")

        # Print summary statistics
        print("\nOptimization Summary:")
        print("-" * 50)
        print(f"Total periods processed: {len(results_df['period'].unique())}")
        print(f"Average number of stocks per period: "
              f"{results_df.groupby('period')['ticker'].count().mean():.2f}")

    except Exception as e:
        logger.error(f"Error during optimization: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()