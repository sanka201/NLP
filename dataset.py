import pandas as pd
import logging
import os

from pandas.errors import EmptyDataError, ParserError

#from backend.dataApp.services.data_validators.data_validator_a import SimpleDataValidator

from validators_nipuni import DataValidator

# need to remove after updating model for dataset to have file_type
def get_file_type(file_path):
    _, file_extension = os.path.splitext(file_path)
    return file_extension


logger = logging.getLogger(__name__)

class Dataset:
    def __init__(self, file_path):
        self.file_path = file_path
        # self.status = status
        self.file_type = get_file_type(file_path)
        self.df = None
        self.error = None
        self.valid = False
        self.load_data()
        
    def load_data(self):
        logger.debug(f"Loading data from {self.file_path} of type {self.file_type}")
        try:
            if self.file_type == ".csv":
                self.df = pd.read_csv(self.file_path)
            elif self.file_type == ".xlsx" or self.file_type == ".xls":
                self.df = pd.read_excel(self.file_path)
            elif self.file_type == ".json":
                self.df = pd.read_json(self.file_path)
            else:
                raise ValueError("Unsupported file type")
            logger.info(f"Data loaded successfully with columns")
        except FileNotFoundError:
            self.error = "File not found."
            logger.error(self.error)
        except EmptyDataError:
            self.error = "No data found in the file"
            logger.error(self.error)
        except ParserError:
            self.error = "Error parsing the file"
            logger.error(self.error)
        except ValueError as ve:
            self.error = str(ve)
            logger.error(self.error)
        except Exception as e:
            self.error = f"An unexpected error occured: {str(e)}"
            logger.error(self.error)

    def validate_data(self):
        if self.df is None:
            logger.error("Data validation failed: no data loaded")
            return ["No data loaded"]

        validator = DataValidator()
        logger.debug("Starting data validation")
        validation_error = validator.validate(self.df)
        
        if isinstance(validation_error, pd.DataFrame):
            logger.info("Data validation passed")
            self.valid = True
            return validation_error
        else:
            logger.warning(f"Validation errors {validation_error}")
            return validation_error
                   
 
