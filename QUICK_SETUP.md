# 🚀 Quick Railway Setup Guide

## Step 1: Create Railway Project
1. Go to: https://railway.app/new
2. Click: **Deploy from GitHub repo**
3. Select: **pazarglobal-whatsapp-bridge**
4. Railway will auto-detect Python

## Step 2: Add Environment Variables

Click on your project → **Variables** tab → **RAW Editor**

Paste this (fill in your actual values):

```env
OPENAI_API_KEY=sk-proj-YOUR_KEY_HERE
OPENAI_WORKFLOW_ID=wf_691884cc7e6081908974fe06852942af0249d08cf5054fdb
TWILIO_ACCOUNT_SID=AC_YOUR_SID_HERE
TWILIO_AUTH_TOKEN=YOUR_TOKEN_HERE
TWILIO_WHATSAPP_NUMBER=+14155238886
```

### Where to get these values:

**OPENAI_API_KEY:**
- https://platform.openai.com/api-keys
- Create new secret key

**OPENAI_WORKFLOW_ID:**
- Already have: `wf_691884cc7e6081908974fe06852942af0249d08cf5054fdb`
- OR check Agent Builder settings

**TWILIO credentials:**
- https://console.twilio.com
- Dashboard → Account Info
- Copy: Account SID & Auth Token

**TWILIO_WHATSAPP_NUMBER:**
- Already set: `+14155238886` (sandbox number)

## Step 3: Deploy

Click **Deploy** → Wait 2-3 minutes

Railway will:
1. Install Python dependencies from `requirements.txt`
2. Run: `uvicorn main:app --host 0.0.0.0 --port $PORT`
3. Assign public URL

## Step 4: Get Your Railway URL

After deployment:
- Click **Settings** → **Generate Domain**
- You'll get: `https://pazarglobal-whatsapp-bridge-production.up.railway.app`

Copy this URL!

## Step 5: Configure Twilio Webhook

1. Go to: https://console.twilio.com/us1/develop/sms/try-it-out/whatsapp-learn
2. Find: **Sandbox settings**
3. Set webhook URL to:
   ```
   https://YOUR-RAILWAY-URL.up.railway.app/webhook/whatsapp
   ```
4. Method: **POST**
5. Click **Save**

## Step 6: Test

### Test 1: Health Check
```bash
curl https://your-railway-url.up.railway.app/health
```

Expected:
```json
{
  "status": "healthy",
  "checks": {
    "openai_key": "configured",
    "twilio_configured": "yes"
  }
}
```

### Test 2: WhatsApp Message
1. Send WhatsApp to: **+1 415 523 8886**
2. First message: `join [your-sandbox-code]`
3. Then: `merhaba`
4. Should get AI response!

## Step 7: Check Logs

Railway dashboard → **Deployments** → **View Logs**

Expected logs:
```
📱 Incoming WhatsApp message from whatsapp:+905551234567: merhaba
🤖 Calling OpenAI Agent Builder workflow: wf_...
✅ Agent response: Merhaba! Size nasıl yardımcı olabilirim?
📤 Sending WhatsApp response to +905551234567
✅ Twilio message sent: SM...
```

---

## ✅ Checklist

- [ ] Railway project created from GitHub repo
- [ ] 5 environment variables added
- [ ] Deployment successful (green checkmark)
- [ ] Domain generated
- [ ] Health endpoint returns 200 OK
- [ ] Twilio webhook URL updated
- [ ] WhatsApp sandbox joined
- [ ] Test message sent and received
- [ ] Logs show successful flow

---

## 🐛 Quick Troubleshooting

**Issue: Build failed**
→ Check Railway logs, likely missing dependency

**Issue: Health check shows "missing" for openai_key**
→ Add OPENAI_API_KEY in Variables tab

**Issue: No WhatsApp response**
→ Check Railway logs, verify webhook URL in Twilio

**Issue: Twilio webhook error**
→ Ensure URL ends with `/webhook/whatsapp`

---

## 🎯 Success Criteria

✅ Health endpoint: 200 OK
✅ WhatsApp message → AI response received
✅ Railway logs show complete flow
✅ No errors in deployment

**Ready to go!** 🚀

---

## 📞 Need Help?

- Railway logs: Check for errors
- Twilio logs: https://console.twilio.com/logs
- GitHub repo: https://github.com/emrahbadas00-lgtm/pazarglobal-whatsapp-bridge
- Deployment guide: See DEPLOYMENT_GUIDE.md for detailed steps
