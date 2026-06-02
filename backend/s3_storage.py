# import json
# import logging
# import os
# import boto3
# from botocore.exceptions import ClientError
# from dotenv import load_dotenv

# load_dotenv()

# logger = logging.getLogger(__name__)

# class S3Manager:
#     def __init__(self):
#         self.bucket_name = os.getenv("S3_BUCKET_NAME").strip() if os.getenv("S3_BUCKET_NAME") else None
#         self.s3_client = boto3.client(
#             "s3",
#             aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
#             aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
#             region_name=os.getenv("AWS_REGION", "ap-southeast-1")
#         )

#         if not self.bucket_name:
#             logger.warning("S3_BUCKET_NAME is not set. S3 operations will fail.")

#     def save_proposal(self, proposal_id: str, data: dict) -> bool:
#         """
#         Saves proposal JSON data to S3.
#         Object Key format: proposals/{proposal_id}.json
#         """
#         if not self.bucket_name:
#             raise ValueError("S3_BUCKET_NAME is not configured in the environment.")

#         object_key = f"proposals/{proposal_id}.json"
        
#         try:
#             json_str = json.dumps(data, indent=4)
#             self.s3_client.put_object(
#                 Bucket=self.bucket_name,
#                 Key=object_key,
#                 Body=json_str,
#                 ContentType="application/json"
#             )
#             logger.info(f"Successfully saved proposal {proposal_id} to S3 bucket {self.bucket_name}")
#             return True
#         except ClientError as e:
#             logger.error(f"Error saving proposal {proposal_id} to S3: {e}")
#             raise e

#     def get_proposal(self, proposal_id: str) -> dict:
#         """
#         Fetches proposal JSON data from S3 by ID.
#         """
#         if not self.bucket_name:
#             raise ValueError("S3_BUCKET_NAME is not configured in the environment.")

#         object_key = f"proposals/{proposal_id}.json"

#         try:
#             response = self.s3_client.get_object(
#                 Bucket=self.bucket_name,
#                 Key=object_key
#             )
#             body = response['Body'].read().decode('utf-8')
#             return json.loads(body)
#         except self.s3_client.exceptions.NoSuchKey:
#             logger.warning(f"Proposal {proposal_id} not found in S3.")
#             return None
#         except ClientError as e:
#             logger.error(f"Error fetching proposal {proposal_id} from S3: {e}")
#             raise e

# # Instantiate a singleton to be used across the app
# s3_manager = S3Manager()
import json
import logging
import os
import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

class S3Manager:
    def __init__(self):
        self.bucket_name = os.getenv("S3_BUCKET_NAME").strip() if os.getenv("S3_BUCKET_NAME") else None
        self.s3_client = boto3.client(
            "s3",
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=os.getenv("AWS_REGION", "ap-southeast-1")
        )

        if not self.bucket_name:
            logger.warning("S3_BUCKET_NAME is not set. S3 operations will fail.")

    def save_proposal(self, proposal_id: str, data: dict) -> bool:
        if not self.bucket_name:
            raise ValueError("S3_BUCKET_NAME is not configured in the environment.")

        object_key = f"proposals/{proposal_id}.json"
        
        try:
            json_str = json.dumps(data, indent=4)
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=object_key,
                Body=json_str,
                ContentType="application/json"
            )
            logger.info(f"Successfully saved proposal {proposal_id} to S3 bucket {self.bucket_name}")
            return True
        except ClientError as e:
            logger.error(f"Error saving proposal {proposal_id} to S3: {e}")
            raise e

    def get_proposal(self, proposal_id: str) -> dict:
        if not self.bucket_name:
            raise ValueError("S3_BUCKET_NAME is not configured in the environment.")

        object_key = f"proposals/{proposal_id}.json"

        try:
            response = self.s3_client.get_object(
                Bucket=self.bucket_name,
                Key=object_key
            )
            body = response['Body'].read().decode('utf-8')
            return json.loads(body)
        except self.s3_client.exceptions.NoSuchKey:
            logger.warning(f"Proposal {proposal_id} not found in S3.")
            return None
        except ClientError as e:
            logger.error(f"Error fetching proposal {proposal_id} from S3: {e}")
            raise e

# Instantiate a singleton to be used across the app
s3_manager = S3Manager()