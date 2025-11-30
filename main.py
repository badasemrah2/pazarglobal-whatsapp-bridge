"""
Pazarglobal WhatsApp Bridge
FastAPI webhook server to bridge WhatsApp (Twilio) with OpenAI Agent Builder
Replaces N8N workflow
"""
import os
from fastapi import FastAPI, Request, HTTPException, Form
from fastapi.responses import Response
import httpx
from twilio.rest import Client
from twilio.twiml.messaging_response import MessagingResponse
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Pazarglobal WhatsApp Bridge")

# Environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_WORKFLOW_ID = os.getenv("OPENAI_WORKFLOW_ID", "wf_691884cc7e6081908974fe06852942af0249d08cf5054fdb")
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_WHATSAPP_NUMBER = os.getenv("TWILIO_WHATSAPP_NUMBER", "+14155238886")

# Initialize Twilio client
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN) if TWILIO_ACCOUNT_SID else None


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Pazarglobal WhatsApp Bridge",
        "version": "1.0.0",
        "twilio_configured": bool(twilio_client),
        "openai_configured": bool(OPENAI_API_KEY)
    }


@app.post("/webhook/whatsapp")
async def whatsapp_webhook(
    Body: str = Form(...),
    From: str = Form(...),
    To: str = Form(None),
    MessageSid: str = Form(None),
):
    """
    Twilio WhatsApp webhook endpoint
    
    Flow:
    1. Receive WhatsApp message from Twilio
    2. Send to OpenAI Agent Builder workflow
    3. Get response from agent
    4. Send back via Twilio WhatsApp
    """
    logger.info(f"📱 Incoming WhatsApp message from {From}: {Body}")
    
    try:
        # Extract phone number (remove 'whatsapp:' prefix)
        phone_number = From.replace('whatsapp:', '')
        user_message = Body
        
        # Step 1: Call OpenAI Agent Builder
        logger.info(f"🤖 Calling OpenAI Agent Builder workflow: {OPENAI_WORKFLOW_ID}")
        agent_response = await call_openai_agent(user_message)
        
        if not agent_response:
            raise HTTPException(status_code=500, detail="No response from OpenAI Agent")
        
        logger.info(f"✅ Agent response: {agent_response[:100]}...")
        
        # Step 2: Send response back via Twilio WhatsApp
        if twilio_client:
            logger.info(f"📤 Sending WhatsApp response to {phone_number}")
            message = twilio_client.messages.create(
                from_=f'whatsapp:{TWILIO_WHATSAPP_NUMBER}',
                body=agent_response,
                to=f'whatsapp:{phone_number}'
            )
            logger.info(f"✅ Twilio message sent: {message.sid}")
        else:
            logger.warning("⚠️ Twilio not configured, response not sent")
        
        # Return TwiML response (Twilio expects this)
        resp = MessagingResponse()
        return Response(content=str(resp), media_type="application/xml")
        
    except Exception as e:
        logger.error(f"❌ Error processing WhatsApp message: {str(e)}")
        logger.exception(e)
        
        # Send error message to user
        resp = MessagingResponse()
        resp.message("Üzgünüm, bir hata oluştu. Lütfen daha sonra tekrar deneyin.")
        return Response(content=str(resp), media_type="application/xml")


async def call_openai_agent(user_input: str) -> str:
    """
    Call OpenAI Agent Builder workflow via API
    
    Args:
        user_input: User's message text
        
    Returns:
        Agent's response text
    """
    if not OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY not configured")
        return "Sistem yapılandırma hatası. Lütfen yönetici ile iletişime geçin."
    
    url = "https://api.openai.com/v1/responses"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
        "OpenAI-Beta": "agentbuilder=v1"
    }
    payload = {
        "workflow_id": OPENAI_WORKFLOW_ID,
        "input": {
            "input_as_text": user_input
        }
    }
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            logger.info(f"🔄 POST {url}")
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            
            data = response.json()
            logger.info(f"📊 OpenAI response status: {response.status_code}")
            
            # Extract output_text from response
            output_text = data.get("output_text", "")
            if not output_text:
                logger.warning(f"⚠️ No output_text in response: {data}")
                return "Üzgünüm, yanıt oluşturamadım."
            
            return output_text
            
    except httpx.HTTPStatusError as e:
        logger.error(f"❌ OpenAI API error: {e.response.status_code} - {e.response.text}")
        return "OpenAI servisi şu anda yanıt vermiyor. Lütfen daha sonra tekrar deneyin."
    except httpx.TimeoutException:
        logger.error("⏱️ OpenAI API timeout")
        return "İstek zaman aşımına uğradı. Lütfen tekrar deneyin."
    except Exception as e:
        logger.error(f"❌ Unexpected error calling OpenAI: {str(e)}")
        return "Beklenmeyen bir hata oluştu."


@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "checks": {
            "openai_key": "configured" if OPENAI_API_KEY else "missing",
            "twilio_configured": "yes" if twilio_client else "no",
            "workflow_id": OPENAI_WORKFLOW_ID
        }
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
