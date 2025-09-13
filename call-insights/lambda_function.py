import json
import boto3
import urllib.request
import urllib.parse
import os
from typing import Dict, Any

def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    AWS Lambda function to invoke the FastAPI call insights endpoint
    
    Handles two types of events:
    1. Direct invocation with custom payload
    2. S3 event notifications (automatic triggers)
    
    Expected event structure for direct invocation:
    {
        "messages": "Your message here",
        "audio_file_key": "path/to/audio/file.mp3",  # Optional: specify specific audio file
        "skip_ingestion": false,  # Optional: skip ingestion step
        "api_url": "http://your-fastapi-url:8000/process"  # Optional, defaults to localhost
    }
    
    S3 event structure (automatic):
    {
        "Records": [
            {
                "s3": {
                    "bucket": {"name": "bucket-name"},
                    "object": {"key": "path/to/file.mp3"}
                }
            }
        ]
    }
    """
    
    try:
        # Check if this is an S3 event notification
        if 'Records' in event and event['Records']:
            # Handle S3 event notification
            s3_record = event['Records'][0]['s3']
            bucket_name = s3_record['bucket']['name']
            file_key = s3_record['object']['key']
            file_size = s3_record['object'].get('size', 0)
            
            # Extract file name for processing
            file_name = file_key.split('/')[-1]
            
            messages = f"Process the newly uploaded audio file: {file_name} (Size: {file_size} bytes)"
            audio_file_key = file_key
            
            print(f"S3 Event detected: Bucket={bucket_name}, Key={file_key}, Size={file_size}")
        else:
            # Handle direct invocation
            messages = event.get('messages', '')
            audio_file_key = event.get('audio_file_key')
            
            if not messages:
                return {
                    'statusCode': 400,
                    'body': json.dumps({
                        'error': 'Missing required field: messages'
                    })
                }
        
        # Get API URL from event or environment variable
        api_url = event.get('api_url', os.environ.get('FASTAPI_URL', 'http://127.0.0.1:8001/process'))
        
        # Prepare the request payload
        payload = {
            "messages": messages,
            "audio_file_key": audio_file_key,  # Use the audio_file_key from above logic
        }
        
        # Remove None values from payload
        payload = {k: v for k, v in payload.items() if v is not None}
        
        print(f"Sending payload to {api_url}: {payload}")
        
        # Make the request to your FastAPI endpoint
        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(
            api_url,
            data=data,
            headers={'Content-Type': 'application/json'},
            method='POST'
        )
        
        try:
            with urllib.request.urlopen(req, timeout=300) as response:
                response_data = json.loads(response.read().decode('utf-8'))
                return {
                    'statusCode': 200,
                    'body': json.dumps({
                        'success': True,
                        'data': response_data,
                        'message': 'Call insights processed successfully'
                    })
                }
        except urllib.error.HTTPError as e:
            error_body = e.read().decode('utf-8')
            return {
                'statusCode': e.code,
                'body': json.dumps({
                    'error': f'API request failed with status {e.code}',
                    'details': error_body
                })
            }
            
    except urllib.error.URLError as e:
        if 'timeout' in str(e).lower():
            return {
                'statusCode': 504,
                'body': json.dumps({
                    'error': 'Request timeout - processing took too long'
                })
            }
        else:
            return {
                'statusCode': 503,
                'body': json.dumps({
                    'error': f'Unable to connect to FastAPI service: {str(e)}'
                })
            }
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': f'Internal server error: {str(e)}'
            })
        }
