
import pandas as pd
import pandera as pa
import os


#Global variables
min_value = 0.0
max_value = 10.0

#importante: inferred_schema = pa.infer_schema(request_table).to_script()
#print(type(request_table))
def main():
    request_table = load_rdbms_requests_table()
    print(value_check(request_table))
    #fixing_values(value_check(request_table))


def value_check(request_table):
    schema = pa.DataFrameSchema({
        "day": pa.Column(
            
nullable=True,
            unique=True,
        ),
        "season": pa.Column(
            dtype="int64",
            checks=[
                pa.Check.greater_than_or_equal_to(min_value=1.0),
                less_than_or_equal_to,
            ],
            nullable=True,
        ),
        "mnth": pa.Column(
            dtype="int64",
            checks=[
                pa.Check.greater_than_or_equal_to(min_value=1.0),
                less_than_or_equal_to,
            ],
            nullable=True,
        ),
        "holiday": pa.Column(
            dtype="int64",
            checks=[
                greater_than_or_equal_to,
                pa.Check.less_than_or_equal_to(max_value=1.0),
            ],
            nullable=True,
        ),
        "weekday": pa.Column(
            dtype="int64",
            checks=[
                greater_than_or_equal_to,
                pa.Check.less_than_or_equal_to(max_value=6.0),
            ],
            nullable=True,
        ),
        "workingday": pa.Column(
            dtype="int64",
            checks=[
                greater_than_or_equal_to,
                pa.Check.less_than_or_equal_to(max_value=1.0),
            ],
            nullable=True,
        ),
        "weathersit": pa.Column(
            dtype="float64",
            checks=[
                pa.Check.greater_than_or_equal_to(min_value=1.0),
                less_than_or_equal_to,
            ],
            nullable=True,
        ),
        "temp": pa.Column(
            dtype="float64",
            checks=[
                pa.Check.greater_than_or_equal_to(min_value=0.0),
                pa.Check.less_than_or_equal_to(max_value=1),
            ],
            nullable=True,
        ),
        "atemp": pa.Column(
            dtype="float64",
            checks=[
                pa.Check.greater_than_or_equal_to(min_value=0.0),
                pa.Check.less_than_or_equal_to(max_value=1),
            ],
            nullable=True,
        ),
        "hum": pa.Column(
            dtype="float64",
            checks=[
                pa.pa.Check.greater_than_or_equal_to(min_value=0.0),
                no_zeros,
                less_than_or_equal_to(),
            ],
            nullable=True,
        ),
        "windspeed": pa.Column(
            dtype="float64",
            checks=[
                pa.Check.greater_than_or_equal_to(min_value=0.0),
                pa.Check.less_than_or_equal_to(max_value=1),
            ],
            nullable=True,
        ),
        "casual": pa.Column(
            dtype="int64",
            nullable=False,
        ),
        "registered": pa.Column(
            dtype="int64",
            nullable=False,
        ),
        "cnt": pa.Column(
            dtype="int64",
            nullable=False,
        ),
    })
    # Validate the dataframe against the schema
    validation_results = schema.validate(request_table)
    return validation_results


# Define a custom validation function that checks for zero values
def no_zeros(Column):
    return (Column != 0).all()

def greater_than_or_equal_to(Column):
    return (Column >= min_value).any()

def less_than_or_equal_to(Column):
    return (Column <= max_value).any()

def fixing_values(validation_result):

    # Check for null values using the `check_null` method
    null_values = validation_result.check_null()

    # Print the row and Column where there is a null value
    for error in null_values.errors:
        print(f"Error in row {error.row}, Column '{error.Column}': {error.value}")

    # Check for zero values using the custom validation function
    zero_values = validation_result.check(lambda x: (x == 0).any())

    # Print the row and Column where there is a value equal to zero
    for error in zero_values.errors:
        print(f"Error in row {error.row}, Column '{error.Column}': {error.value}")






def load_rdbms_requests_table():
    module_path = os.path.dirname(__file__)
    filename = os.path.join(module_path, "../raw-data/requests_table.csv")
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File {filename} not found")
    data = pd.read_csv(filename, sep=",")
    return data


if __name__ == "__main__":
    main()