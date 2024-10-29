# services/data_validator.py

import os
import re
import logging
from abc import ABC, abstractmethod
import unittest
import pandas as pd

logger = logging.getLogger(__name__)


class DataValidatorinterface(ABC):
    @abstractmethod
    def validate(self, file_url):
        pass


class DataValidator(DataValidatorinterface):
    """This class validates bla..."""

    def __init__(self):
        self.required_columns = [
            "LatestPartNbr",
            "PartType",
            "RootPartType",
            "PrintGroup",
            "MSTName",
            "MSTCity",
            "MSTState",
            "MSTCountry",
            "Type",
            "UnitsOrdered",
        ]
        self.special_char_pattern = re.compile(r"[^a-zA-Z0-9_\s.]")

    def check_required_columns(self, df):
        missing_columns = [
            col for col in self.required_columns if col not in df.columns
        ]
        if missing_columns:
            message = f"Missing required columns: {', '.join(missing_columns)}"
            logger.error(message)
            return message
        return None

    def check_nan_values(self, df):
        nan_info = df.isna().sum()
        if nan_info.any():
            nan_details = nan_info[nan_info > 0]
            message = f"NaN values found in the following columns:\n{nan_details}"
            logger.error(message)
            return message
        return None

    def check_special_characters_in_entries(self, df):
        for column in df[
            ["LatestPartNbr", "MSTState", "MSTCountry", "Type", "UnitsOrdered"]
        ]:
            special_chars = df[column].apply(
                lambda x: bool(self.special_char_pattern.search(str(x)))
            )
            if special_chars.any():
                rows_with_special_chars = df[special_chars].index.tolist()
                message = f"Special characters found in column '{column}' at rows: {rows_with_special_chars}"
                logger.error(message)
                return message
        return None

    def check_duplicated_rows(self, df):
        duplicated = df[df.duplicated()]
        if not duplicated.empty:
            message = f"Duplicated rows found:\n{duplicated}"
            logger.error(message)
            return message
        return None

    def validate(self, data_inp):
        df = data_inp
        print(df)
        checks = [
            self.check_required_columns,
            self.check_nan_values,
            self.check_special_characters_in_entries,
            self.check_duplicated_rows,
        ]

        for check in checks:
            error_message = check(df)
            if error_message:
                return None, error_message
        logger.info("Validation successful : no errors found")
        return df, "File validation successful. Proceeding with analysis."


# Unit test section (more to be added)
class TestDataValidator(unittest.TestCase):

    def setUp(self):
        self.validator = DataValidator()
        self.test_file_path = "/home/sanka/volttron/annual_forecast_to_MST_10rows.xlsx"
        if os.path.isfile(self.test_file_path):
            self.df = pd.read_excel(self.test_file_path)
        else:
            self.df = None

    def test_file_exists(self):
        self.assertTrue(
            os.path.isfile(self.test_file_path),
            f"File '{self.test_file_path}' does not exist.",
        )

    def test_validation(self):
        if self.df is not None:
            self.assertIsNotNone(self.validator.validate(self.df))
        else:
            self.fail(f"File '{self.test_file_path}' could not be loaded.")


if __name__ == "__main__":
    # Run unit tests
    unittest.main()