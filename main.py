"""
Pazarglobal WhatsApp Bridge
FastAPI webhook server to bridge WhatsApp (Twilio) with Agent Backend (OpenAI Agents SDK)
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
AGENT_BACKEND_URL = os.getenv("AGENT_BACKEND_URL", "https://pazarglobal-agent-backend-production.up.railway.app")
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
        "version": "3.0.0",
        "api_type": "Agent Backend (OpenAI Agents SDK)",
        "twilio_configured": bool(twilio_client),
        "agent_backend_url": AGENT_BACKEND_URL
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
    2. Send to Agent Backend (OpenAI Agents SDK with MCP tools)
    3. Get response from agent
    4. Send back via Twilio WhatsApp
    """
    logger.info(f"📱 Incoming WhatsApp message from {From}: {Body}")
    
    try:
        # Extract phone number (remove 'whatsapp:' prefix)
        phone_number = From.replace('whatsapp:', '')
        user_message = Body
        
        # Step 1: Call Agent Backend
        logger.info(f"🤖 Calling Agent Backend: {AGENT_BACKEND_URL}")
        agent_response = await call_agent_backend(user_message, phone_number)
        
        if not agent_response:
            raise HTTPException(status_code=500, detail="No response from Agent Backend")
        
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


async def call_agent_backend(user_input: str, user_id: str) -> str:
    """
    Call Agent Backend (OpenAI Agents SDK with MCP tools)
    
    Args:
        user_input: User's message text
        user_id: User identifier (phone number)
        
    Returns:
        Agent's response text
    """
    if not AGENT_BACKEND_URL:
        logger.error("AGENT_BACKEND_URL not configured")
        return "Sistem yapılandırma hatası. Lütfen yönetici ile iletişime geçin."
    
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            # Call agent backend endpoint
            logger.info(f"🚀 Calling Agent Backend: {AGENT_BACKEND_URL}/agent/run")
            
            payload = {
                "user_id": user_id,
                "message": user_input,
                "conversation_history": []  # Can be extended to maintain conversation history
            }
            
            response = await client.post(
                f"{AGENT_BACKEND_URL}/agent/run",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            
            result = response.json()
            logger.info(f"✅ Agent Backend response received")
            logger.info(f"   Intent: {result.get('intent', 'unknown')}")
            logger.info(f"   Success: {result.get('success', False)}")
            
            if not result.get("success"):
                logger.error(f"⚠️ Agent Backend returned success=false")
                return "İşlem başarısız oldu. Lütfen tekrar deneyin."
            
            response_text = result.get("response", "")
            if not response_text:
                logger.error("⚠️ Empty response from Agent Backend")
                return "Boş yanıt alındı. Lütfen tekrar deneyin."
            
            logger.info(f"✅ Response text: {response_text[:100]}...")
            return response_text
            
    except httpx.HTTPStatusError as e:
        logger.error(f"❌ Agent Backend HTTP error: {e.response.status_code}")
        try:
            error_detail = e.response.json()
            logger.error(f"   Error detail: {error_detail}")
        except:
            logger.error(f"   Response text: {e.response.text}")
        return "Agent servisi şu anda yanıt vermiyor. Lütfen daha sonra tekrar deneyin."
    except httpx.TimeoutException:
        logger.error("⏱️ Agent Backend timeout (120s)")
        return "İstek zaman aşımına uğradı. Lütfen tekrar deneyin."
    except Exception as e:
        logger.error(f"❌ Unexpected error calling Agent Backend: {str(e)}")
        logger.exception(e)
        return "Beklenmeyen bir hata oluştu."


@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "checks": {
            "agent_backend_url": AGENT_BACKEND_URL,
            "twilio_configured": "yes" if twilio_client else "no"
        }
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
