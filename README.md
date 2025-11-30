# Pazarglobal WhatsApp Bridge 📱

FastAPI webhook server that bridges Twilio WhatsApp with OpenAI Agent Builder.

## 🎯 Purpose

Replaces N8N workflow for WhatsApp integration. Enables WhatsApp users to interact with Pazarglobal AI agents.

## 🏗️ Architecture

```
WhatsApp User
    ↓
Twilio WhatsApp API
    ↓
[THIS SERVER] /webhook/whatsapp
    ↓
OpenAI Agent Builder (Workflow API)
    ↓
[THIS SERVER] (receives response)
    ↓
Twilio WhatsApp API
    ↓
WhatsApp User
```

## 🚀 Features

- ✅ Twilio WhatsApp webhook handler
- ✅ OpenAI Agent Builder integration
- ✅ Automatic response routing
- ✅ Error handling & logging
- ✅ Health check endpoints

## 📦 Installation

```bash
pip install -r requirements.txt
```

## 🔧 Environment Variables

Required:
```env
OPENAI_API_KEY=sk-...
OPENAI_WORKFLOW_ID=wf_...
TWILIO_ACCOUNT_SID=AC...
TWILIO_AUTH_TOKEN=...
TWILIO_WHATSAPP_NUMBER=+14155238886
PORT=8080
```

## 🏃 Running Locally

```bash
python main.py
```

Server will start on `http://localhost:8080`

## 🌐 Endpoints

### `POST /webhook/whatsapp`
Twilio WhatsApp webhook endpoint.

**Expected Form Data:**
- `Body`: Message text
- `From`: Sender phone number (whatsapp:+1234567890)
- `To`: Recipient number
- `MessageSid`: Twilio message ID

### `GET /`
Health check and status.

### `GET /health`
Detailed health check with configuration status.

## 📡 Railway Deployment

1. **Create new Railway project**
2. **Connect this GitHub repo**
3. **Set environment variables** in Railway dashboard
4. **Deploy** - Railway auto-detects Python and installs dependencies

### Railway Configuration

**Start Command:**
```bash
uvicorn main:app --host 0.0.0.0 --port $PORT
```

**Environment Variables** (set in Railway):
```
OPENAI_API_KEY=your_openai_key
OPENAI_WORKFLOW_ID=wf_691884cc7e6081908974fe06852942af0249d08cf5054fdb
TWILIO_ACCOUNT_SID=your_twilio_sid
TWILIO_AUTH_TOKEN=your_twilio_token
TWILIO_WHATSAPP_NUMBER=+14155238886
```

## 🔗 Twilio Setup

1. Go to Twilio Console → Messaging → Try it out → WhatsApp
2. Set webhook URL: `https://your-railway-url.up.railway.app/webhook/whatsapp`
3. Method: POST
4. Test with WhatsApp Sandbox number

## 🧪 Testing

### Local Testing (without Twilio)
```bash
curl -X POST http://localhost:8080/webhook/whatsapp \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "Body=merhaba&From=whatsapp:+905551234567"
```

### Health Check
```bash
curl http://localhost:8080/health
```

Expected response:
```json
{
  "status": "healthy",
  "checks": {
    "openai_key": "configured",
    "twilio_configured": "yes",
    "workflow_id": "wf_..."
  }
}
```

## 📊 Flow Details

### 1. Receive WhatsApp Message
```
User sends: "laptop aramak istiyorum"
Twilio POST to /webhook/whatsapp
```

### 2. Call OpenAI Agent
```python
POST https://api.openai.com/v1/responses
{
  "workflow_id": "wf_...",
  "input": {
    "input_as_text": "laptop aramak istiyorum"
  }
}
```

### 3. Get Agent Response
```json
{
  "output_text": "Laptop aramanıza yardımcı olabilirim! Hangi fiyat aralığında..."
}
```

### 4. Send WhatsApp Response
```python
twilio_client.messages.create(
  from_='whatsapp:+14155238886',
  body='Laptop aramanıza yardımcı olabilirim!...',
  to='whatsapp:+905551234567'
)
```

## 🔒 Security Notes

- ✅ Twilio webhook signature validation (recommended to add)
- ✅ HTTPS only in production
- ✅ Environment variables for secrets
- ✅ Error messages sanitized for users

## 🐛 Troubleshooting

### Issue: "OPENAI_API_KEY not configured"
**Solution:** Set environment variable in Railway

### Issue: "Twilio not configured"
**Solution:** Set TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN

### Issue: "No response from OpenAI Agent"
**Solution:** Check workflow_id is correct and workflow is active

### Issue: Twilio webhook not triggering
**Solution:** 
1. Verify Railway URL is accessible
2. Check Twilio webhook settings
3. Ensure URL ends with `/webhook/whatsapp`

## 📝 Logs

Railway logs show:
```
📱 Incoming WhatsApp message from whatsapp:+905551234567: merhaba
🤖 Calling OpenAI Agent Builder workflow: wf_...
🔄 POST https://api.openai.com/v1/responses
📊 OpenAI response status: 200
✅ Agent response: Merhaba! Size nasıl yardımcı olabilirim?
📤 Sending WhatsApp response to +905551234567
✅ Twilio message sent: SM...
```

## 🔄 Differences from N8N Workflow

| Feature | N8N | This Server |
|---------|-----|-------------|
| Hosting | N8N Cloud | Railway |
| Language | Visual nodes | Python |
| Cost | Paid subscription | Free tier |
| Customization | Limited | Full control |
| Debugging | Visual | Logs |
| Deployment | N8N platform | GitHub + Railway |

## 🚦 Status Codes

- `200` - Success, TwiML response returned
- `500` - Internal server error
- `503` - OpenAI API unavailable

## 📞 Support

If issues persist:
1. Check Railway logs
2. Test `/health` endpoint
3. Verify Twilio webhook configuration
4. Check OpenAI workflow is active

## 🔗 Related Projects

- [pazarglobal_mcp](https://github.com/emrahbadas00-lgtm/Pazarglobal) - MCP Server with tools
- OpenAI Agent Builder - Agent workflow configuration

## 📄 License

MIT

## 🤝 Contributing

This is part of the Pazarglobal project. For WhatsApp bridge issues, open an issue in this repo.
