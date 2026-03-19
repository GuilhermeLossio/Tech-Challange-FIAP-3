import boto3
import pandas as pd
import io
from urllib.parse import unquote_plus

def convert_csv_to_parquet(bucket_name):
    """
    Converts all CSV files in a given S3 bucket to Parquet format.

    For each CSV file found in the bucket, this function will:
    1. Read the CSV file into a pandas DataFrame.
    2. Write the DataFrame to a Parquet file in-memory.
    3. Upload the Parquet file to the same S3 bucket with the same name,
       but with a .parquet extension.
    4. Delete the original CSV file from the bucket.

    Args:
        bucket_name (str): The name of the S3 bucket to process.
    """
    s3_client = boto3.client('s3')
    s3_resource = boto3.resource('s3')

    try:
        # List all objects in the bucket
        response = s3_client.list_objects_v2(Bucket=bucket_name)
        if 'Contents' not in response:
            print(f"No objects found in bucket '{bucket_name}'.")
            return

        print(f"Found objects in bucket '{bucket_name}'. Checking for CSV files...")

        for obj in response['Contents']:
            key = obj['Key']
            # unquote_plus is used to handle spaces or other special characters in file names
            key = unquote_plus(key)

            if key.endswith('.csv'):
                print(f"Processing file: {key}")
                try:
                    # Get the CSV file object from S3
                    csv_obj = s3_client.get_object(Bucket=bucket_name, Key=key)
                    
                    # Read CSV data into a pandas DataFrame
                    df = pd.read_csv(io.BytesIO(csv_obj['Body'].read()))

                    # Create the new Parquet file name
                    parquet_key = key.rsplit('.', 1)[0] + '.parquet'

                    # Write DataFrame to a Parquet buffer
                    parquet_buffer = io.BytesIO()
                    df.to_parquet(parquet_buffer, index=False)
                    
                    # Reset buffer position to the beginning
                    parquet_buffer.seek(0)

                    # Upload the Parquet file to S3
                    s3_client.put_object(Bucket=bucket_name, Key=parquet_key, Body=parquet_buffer.read())
                    print(f"Successfully converted '{key}' to '{parquet_key}'")

                    # Delete the original CSV file
                    s3_client.delete_object(Bucket=bucket_name, Key=key)
                    print(f"Successfully deleted original file: '{key}'")

                except Exception as e:
                    print(f"Error processing file {key}: {e}")

        print("Conversion process finished.")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    # The user mentioned the bucket name in the prompt.
    BUCKET = 'flight-advisor-fiap3'
    print(f"Starting CSV to Parquet conversion for bucket: '{BUCKET}'")
    convert_csv_to_parquet(BUCKET)
