import json
import urllib.request
import os
from typing import Dict, Any

def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    AWS Lambda function to invoke the LangGraph server directly
    
    S3 event structure:
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
            
            file_name = file_key.split('/')[-1]
            messages = f"Process the newly uploaded audio file: {file_name}"
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
        
        # Use LangGraph server endpoint
        langgraph_url = os.environ.get('LANGGRAPH_URL', 'http://127.0.0.1:2024/threads')
        
        # Prepare the request payload for LangGraph
        payload = {
            "input": {
                "messages": [{"role": "user", "content": messages}],
                "audio_file_key": audio_file_key
            }
        }
        
        # Remove None values
        if payload["input"]["audio_file_key"] is None:
            del payload["input"]["audio_file_key"]
        
        print(f"Sending payload to {langgraph_url}: {payload}")
        
        # Make the request to LangGraph
        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(
            langgraph_url,
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
                    'error': f'LangGraph request failed with status {e.code}',
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
                    'error': f'Unable to connect to LangGraph service: {str(e)}'
                })
            }
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': f'Internal server error: {str(e)}'
            })
        }
