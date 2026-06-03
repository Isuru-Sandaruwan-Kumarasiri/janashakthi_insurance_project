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



#=================


import json
import logging
import os
from datetime import datetime
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

    def _build_key(self, proposal_id: str) -> str:
        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
        return f"proposals/{proposal_id}/{timestamp}.json"

    def save_proposal(self, proposal_id: str, data: dict) -> bool:
        if not self.bucket_name:
            raise ValueError("S3_BUCKET_NAME is not configured in the environment.")

        object_key = self._build_key(proposal_id)

        try:
            json_str = json.dumps(data, indent=4)
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=object_key,
                Body=json_str,
                ContentType="application/json"
            )
            logger.info(f"Successfully saved proposal {proposal_id} to S3 at {object_key}")
            return True
        except ClientError as e:
            logger.error(f"Error saving proposal {proposal_id} to S3: {e}")
            raise e

    def get_proposal(self, proposal_id: str, timestamp: str = None) -> dict:
        """
        Fetch a proposal by ID.
        - If timestamp is provided (format: YYYYMMDDTHHmmSS), fetch that specific version.
        - If not provided, fetch the latest version.
        """
        if not self.bucket_name:
            raise ValueError("S3_BUCKET_NAME is not configured in the environment.")

        if timestamp:
            object_key = f"proposals/{proposal_id}/{timestamp}.json"
        else:
            object_key = self._get_latest_key(proposal_id)
            if not object_key:
                logger.warning(f"No versions found for proposal {proposal_id}.")
                return None

        try:
            response = self.s3_client.get_object(
                Bucket=self.bucket_name,
                Key=object_key
            )
            body = response['Body'].read().decode('utf-8')
            return json.loads(body)
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                logger.warning(f"Proposal {proposal_id} not found at key {object_key}.")
                return None
            logger.error(f"Error fetching proposal {proposal_id} from S3: {e}")
            raise e

    def _get_latest_key(self, proposal_id: str) -> str | None:
        """List all versions for a proposal and return the key with the latest timestamp."""
        prefix = f"proposals/{proposal_id}/"
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix
            )
            objects = response.get("Contents", [])
            if not objects:
                return None
            # Keys are timestamp-named, so lexicographic sort gives the latest
            latest = max(objects, key=lambda obj: obj["Key"])
            return latest["Key"]
        except ClientError as e:
            logger.error(f"Error listing versions for proposal {proposal_id}: {e}")
            raise e

    def list_versions(self, proposal_id: str) -> list[str]:
        """Return all saved timestamps for a given proposal ID, sorted oldest to newest."""
        prefix = f"proposals/{proposal_id}/"
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix
            )
            objects = response.get("Contents", [])
            # Extract just the timestamp portion from each key
            timestamps = [
                obj["Key"].replace(prefix, "").replace(".json", "")
                for obj in sorted(objects, key=lambda o: o["Key"])
            ]
            return timestamps
        except ClientError as e:
            logger.error(f"Error listing versions for proposal {proposal_id}: {e}")
            raise e

# Instantiate a singleton to be used across the app
s3_manager = S3Manager()