
from datetime import datetime
from dateutil.relativedelta import relativedelta
from typing import Union


class dateUtil:

    @staticmethod
    def validate_start_date(start_date: Union[str, datetime]) -> str:
        """
        Validate and format the start date input.

        Args:
            start_date: String in format 'YYYY-MM-DD' or datetime object

        Returns:
            str: Validated and formatted date string in 'YYYY-MM-DD' format

        Raises:
            ValueError: If date format is invalid or date is in the future
        """
        try:
            # Convert string to datetime if necessary
            if isinstance(start_date, str):
                date_obj = datetime.strptime(start_date, '%Y-%m-%d')
            elif isinstance(start_date, datetime):
                date_obj = start_date
            else:
                raise ValueError("Start date must be a string 'YYYY-MM-DD' or datetime object")

            # Validate date is not in the future
            if date_obj > datetime.now():
                raise ValueError("Start date cannot be in the future")

            # Validate date is not too old (e.g., more than 10 years ago)
            min_date = datetime.now() - relativedelta(years=10)
            if date_obj < min_date:
                raise ValueError("Start date cannot be more than 10 years ago")

            # Return formatted string
            return date_obj.strftime('%Y-%m-%d')

        except ValueError as e:
            raise ValueError(f"Invalid start date: {str(e)}")

    @staticmethod
    def validate_granularity(granularity: Union[int, str]) -> int:
        """
        Validate the granularity input.

        Args:
            granularity: Integer representing the number of days

        Returns:
            int: Validated granularity value

        Raises:
            ValueError: If granularity is invalid
        """
        try:
            granularity = int(granularity)
            if granularity < 1:
                raise ValueError("Granularity must be positive")
            if granularity > 365:
                raise ValueError("Granularity must be less than or equal to 365")
            return granularity
        except (ValueError, TypeError):
            raise ValueError("Granularity must be a valid integer")