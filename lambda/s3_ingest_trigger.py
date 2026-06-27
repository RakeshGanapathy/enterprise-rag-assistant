"""
AWS Lambda function — triggered by S3 event notifications.

Deploy this in AWS Lambda. It does NOT live inside the RAG API.

Trigger setup (CloudFormation / Terraform / Console):
  S3 bucket -> Properties -> Event notifications
    Event types: s3:ObjectCreated:*, s3:ObjectRemoved:*
    Destination: this Lambda function

IAM permissions this Lambda needs:
  s3:GetObject          on the document bucket
  s3:GeneratePresignedUrl  (implicit with GetObject)

The RAG API needs no S3 credentials — Lambda passes a presigned URL.

Environment variables (set in Lambda config):
  RAG_API_URL    e.g. https://rag.internal.company.com
  RAG_API_KEY    shared secret for the /documents/ingest-s3 endpoint
"""
import json
import os
import urllib.request

import boto3

RAG_API_URL = os.environ["RAG_API_URL"]
RAG_API_KEY = os.environ.get("RAG_API_KEY", "")
PRESIGNED_URL_EXPIRY = 300   # seconds — long enough for the RAG API to download


def handler(event, context):
    s3_client = boto3.client("s3")
    results = []

    for record in event.get("Records", []):
        bucket = record["s3"]["bucket"]["name"]
        key    = record["s3"]["object"]["key"]
        event_name = record["eventName"]   # e.g. "ObjectCreated:Put", "ObjectRemoved:Delete"

        event_type = "deleted" if "ObjectRemoved" in event_name else "created"

        # Read department and access_level from S3 object tags
        # These are set by the document uploader, not inferred from the filename
        department   = None
        access_level = None

        if event_type != "deleted":
            try:
                tags_response = s3_client.get_object_tagging(Bucket=bucket, Key=key)
                tags = {t["Key"]: t["Value"] for t in tags_response.get("TagSet", [])}
                department   = tags.get("department")
                access_level = tags.get("access_level")
            except Exception as e:
                print(f"Warning: could not read tags for {key}: {e}")

        # Generate presigned URL — lets the RAG API download the file
        # without needing its own S3 credentials
        presigned_url = None
        if event_type != "deleted":
            presigned_url = s3_client.generate_presigned_url(
                "get_object",
                Params={"Bucket": bucket, "Key": key},
                ExpiresIn=PRESIGNED_URL_EXPIRY,
            )

        # Call the RAG API
        payload = {
            "bucket": bucket,
            "key": key,
            "event_type": event_type,
            "department": department,
            "access_level": access_level,
            "presigned_url": presigned_url,
        }

        response = _call_rag_api("/documents/ingest-s3", payload)
        results.append({
            "key": key,
            "event_type": event_type,
            "job_id": response.get("job_id"),
            "status": response.get("status"),
        })
        print(f"Queued: {key} -> job_id={response.get('job_id')}")

    return {"statusCode": 200, "body": json.dumps(results)}


def _call_rag_api(path: str, payload: dict) -> dict:
    url = RAG_API_URL.rstrip("/") + path
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "X-Api-Key": RAG_API_KEY,
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read())
