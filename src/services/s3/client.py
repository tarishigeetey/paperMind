"""
S3Client — durable PDF storage (Episode 10.1).

Java analogy: this plays the same role as a Spring @Repository wrapping
an AWS SDK client — S3Client is boto3's version of the AWS SDK's
S3Client/AmazonS3 interface. We wrap it so the rest of the app depends
on OUR small interface (upload_file/download_file/object_exists), not
on boto3 directly — same reason you'd never sprinkle raw JDBC calls
through a Java codebase instead of going through a Repository.
"""

import logging
from pathlib import Path

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from src.config import Settings
from src.exceptions import S3DownloadError, S3UploadError

logger = logging.getLogger(__name__)


class S3Client:
    """Thin wrapper around boto3's S3 client, scoped to one bucket."""

    def __init__(self, settings: Settings):
        self.bucket_name = settings.s3.bucket_name

        # Java analogy: AmazonS3ClientBuilder.standard().withRegion(...).withCredentials(...).build()
        # `or None` matters here — passing empty strings to boto3 makes it
        # try (and fail) to authenticate with blank credentials instead of
        # falling back to its normal credential chain (env vars, ~/.aws/credentials,
        # an instance/task role, etc.) if we ever stop passing keys explicitly.
        self._client = boto3.client(
            "s3",
            region_name=settings.s3.region,
            aws_access_key_id=settings.s3.access_key_id or None,
            aws_secret_access_key=settings.s3.secret_access_key or None,
        )

    def object_exists(self, key: str) -> bool:
        """Check whether `key` exists in the bucket, without downloading it.

        head_object = an HTTP HEAD request — same idea as Java's
        HttpURLConnection.setRequestMethod("HEAD"): get metadata/existence
        without paying for the full object body.
        """
        try:
            self._client.head_object(Bucket=self.bucket_name, Key=key)
            return True
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            if error_code in ("404", "NoSuchKey"):
                return False
            raise S3DownloadError(f"Error checking s3://{self.bucket_name}/{key}: {e}")

    def upload_file(self, local_path: Path, key: str) -> None:
        """Upload a local file to S3 under `key`. Overwrites if it already exists."""
        try:
            self._client.upload_file(str(local_path), self.bucket_name, key)
            logger.info(f"Uploaded {local_path.name} to s3://{self.bucket_name}/{key}")
        except (BotoCoreError, ClientError) as e:
            raise S3UploadError(f"Failed to upload {local_path} to s3://{self.bucket_name}/{key}: {e}")

    def download_file(self, key: str, local_path: Path) -> None:
        """Download `key` from S3 to a local path, creating parent dirs as needed."""
        try:
            local_path.parent.mkdir(parents=True, exist_ok=True)
            self._client.download_file(self.bucket_name, key, str(local_path))
            logger.info(f"Downloaded s3://{self.bucket_name}/{key} to {local_path}")
        except (BotoCoreError, ClientError) as e:
            raise S3DownloadError(f"Failed to download s3://{self.bucket_name}/{key} to {local_path}: {e}")

    def health_check(self) -> bool:
        """Verify the bucket is reachable with the configured credentials."""
        try:
            self._client.head_bucket(Bucket=self.bucket_name)
            return True
        except (BotoCoreError, ClientError) as e:
            logger.warning(f"S3 health check failed for bucket {self.bucket_name}: {e}")
            return False
