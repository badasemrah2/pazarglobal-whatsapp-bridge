"""
Test OpenAI Agent Builder API call
"""
import os
import asyncio
import httpx
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_WORKFLOW_ID = os.getenv("OPENAI_WORKFLOW_ID")

async def test_agent_builder():
    """Test Agent Builder API"""
    
    url = "https://api.openai.com/v1/responses"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
        "OpenAI-Beta": "agentbuilder=v1"
    }
    payload = {
        "workflow_id": OPENAI_WORKFLOW_ID,
        "input": {
            "input_as_text": "merhaba test"
        }
    }
    
    print(f"🔄 Testing Agent Builder API...")
    print(f"URL: {url}")
    print(f"Workflow ID: {OPENAI_WORKFLOW_ID}")
    print(f"Payload: {payload}\n")
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, headers=headers, json=payload)
            
            print(f"Status: {response.status_code}")
            print(f"Response: {response.text}\n")
            
            if response.is_success:
                data = response.json()
                print(f"✅ Success!")
                print(f"Output: {data.get('output_text', 'No output_text')}")
            else:
                print(f"❌ Error: {response.status_code}")
                print(f"Details: {response.text}")
                
    except Exception as e:
        print(f"❌ Exception: {type(e).__name__}: {str(e)}")

if __name__ == "__main__":
    asyncio.run(test_agent_builder())
