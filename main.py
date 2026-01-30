"""
Pazarglobal WhatsApp Bridge
FastAPI webhook server to bridge WhatsApp (Twilio) with Agent Backend (OpenAI Agents SDK)
Replaces N8N workflow
"""
import ast
import io
import json
import os
import uuid
import re
from fastapi import FastAPI, Request, HTTPException, Form
from fastapi.responses import Response
import httpx
from twilio.rest import Client
from twilio.twiml.messaging_response import MessagingResponse
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from PIL import Image
from redis_helper import redis_client  # Redis client for persistent storage

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Pazarglobal WhatsApp Bridge")

# Environment variables
AGENT_BACKEND_URL = os.getenv("AGENT_BACKEND_URL", "https://pazarglobal-agent-production.up.railway.app")

# Edge Function URL (Traffic Controller)
# IMPORTANT: Must point to your Supabase project, otherwise WhatsApp traffic bypasses the security gate.
EDGE_FUNCTION_URL = os.getenv("EDGE_FUNCTION_URL", "https://snovwbffwvmkgjulrtsm.supabase.co/functions/v1/whatsapp-traffic-controller")

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_WHATSAPP_NUMBER = os.getenv("TWILIO_WHATSAPP_NUMBER", "+14155238886")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
SUPABASE_STORAGE_BUCKET = os.getenv("SUPABASE_STORAGE_BUCKET", "product-images")

# Initialize Twilio client
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN) if TWILIO_ACCOUNT_SID else None

# ========== CONVERSATION HISTORY CACHE ==========
# Now using Redis for persistent storage with in-memory fallback
# In-memory storage (legacy fallback when Redis unavailable):
conversation_store: Dict[str, dict] = {}
CONVERSATION_TIMEOUT_MINUTES = 30  # Clear conversations after 30 minutes of inactivity
MAX_MEDIA_BYTES = 10 * 1024 * 1024  # 10 MB limit
MAX_MEDIA_PER_MESSAGE = 3  # Avoid WhatsApp bulk; keep under total size limits

# If true, WhatsApp bridge will render numbered listing detail from its in-memory cache
# without calling backend. Default is False to keep WhatsApp/WebChat behavior consistent.
WHATSAPP_LOCAL_DETAIL_SHORTCIRCUIT = os.getenv("WHATSAPP_LOCAL_DETAIL_SHORTCIRCUIT", "false").lower() in ("1", "true", "yes")


def _extract_last_media_context(history: List[dict]) -> tuple[Optional[str], List[str]]:
    """Fetch ONLY the most recent draft id and media paths from system notes."""
    draft_id = None
    all_media_paths: List[str] = []

    # Only look at the most recent SYSTEM_MEDIA_NOTE (not all historical ones)
    for msg in reversed(history or []):
        text = msg.get("content") if isinstance(msg, dict) else None
        if not isinstance(text, str):
            continue
        if "[SYSTEM_MEDIA_NOTE]" not in text:
            continue

        # Found the most recent media note - extract and stop
        if "DRAFT_LISTING_ID=" in text:
            draft_id = text.split("DRAFT_LISTING_ID=", 1)[1].split("|")[0].strip()

        if "MEDIA_PATHS=" in text:
            raw_paths = text.split("MEDIA_PATHS=", 1)[1].split("|")[0].strip()
            try:
                parsed = ast.literal_eval(raw_paths)
                if isinstance(parsed, list):
                    all_media_paths = [p for p in parsed if isinstance(p, str)]
            except Exception:
                pass
        
        # IMPORTANT: Stop after first (most recent) media note
        break

    return draft_id, all_media_paths


def _sanitize_user_id(user_id: str) -> str:
    # Twilio phone comes as +90..., remove plus and spaces for path safety
    return (user_id or "unknown").replace("+", "").replace(" ", "")


def _build_storage_path(user_id: str, listing_uuid: str, media_type: Optional[str]) -> str:
    ext = (media_type or "image/jpeg").split("/")[-1] or "jpg"
    return f"{_sanitize_user_id(user_id)}/{listing_uuid}/{uuid.uuid4()}.{ext}"


async def download_media(media_url: str, media_type: Optional[str], message_sid: Optional[str], media_sid: Optional[str]) -> Optional[tuple[bytes, str]]:
    if not media_url:
        return None
    if not TWILIO_ACCOUNT_SID or not TWILIO_AUTH_TOKEN:
        logger.warning("Twilio credentials missing, cannot fetch media")
        return None
    try:
        logger.info(f"📥 Downloading media from: {media_url[:80]}...")
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            resp = await client.get(media_url, auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN))
        
        logger.info(f"📊 Download response: status={resp.status_code}, content-type={resp.headers.get('Content-Type')}")
        
        if resp.status_code == 404 and twilio_client and message_sid and media_sid:
            logger.warning(f"⚠️ Direct URL returned 404, trying Twilio API fallback...")
            try:
                media_obj = twilio_client.messages(message_sid).media(media_sid).fetch()
                # Twilio returns uri like /2010-04-01/Accounts/AC.../Messages/MM.../Media/ME....json
                if media_obj.uri:
                    fallback_url = f"https://api.twilio.com{media_obj.uri.replace('.json','')}"
                else:
                    raise ValueError("Media object has no URI")
                logger.info(f"🔄 Fallback URL: {fallback_url}")
                async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
                    resp = await client.get(fallback_url, auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN))
                logger.info(f"📊 Fallback response: status={resp.status_code}")
            except Exception as tw_err:
                logger.error(f"❌ Twilio fallback media fetch failed: {tw_err}")
                return None

        if not resp.is_success:
            logger.error(f"❌ Failed to download media: status={resp.status_code}, body={resp.text[:200]}")
            return None
        
        content_type = resp.headers.get("Content-Type", media_type or "")
        if not content_type.startswith("image/"):
            logger.warning(f"⚠️ Blocked non-image media: {content_type}")
            return None
        
        content = resp.content
        if content and len(content) > MAX_MEDIA_BYTES:
            logger.warning(f"⚠️ Media too large ({len(content)} bytes), skipping upload")
            return None
        
        logger.info(f"✅ Media downloaded successfully: {len(content)} bytes, type={content_type}")
        return content, content_type
    except Exception as e:
        logger.error(f"❌ Error downloading media: {e}", exc_info=True)
        return None


def _compress_image(content: bytes, media_type: Optional[str]) -> Optional[tuple[bytes, str]]:
    """Downsize and recompress image to keep WhatsApp-friendly size."""
    try:
        img = Image.open(io.BytesIO(content))
        img = img.convert("RGB")  # Ensure JPEG-compatible

        max_side = 1600
        w, h = img.size
        if max(w, h) > max_side:
            ratio = max_side / float(max(w, h))
            # Use Resampling.LANCZOS for newer Pillow, fallback to LANCZOS for older versions
            try:
                from PIL.Image import Resampling
                img = img.resize((int(w * ratio), int(h * ratio)), Resampling.LANCZOS)
            except ImportError:
                img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)

        target_bytes = 900_000  # ~0.9 MB target to stay well under Twilio limits
        quality = 85
        min_quality = 50
        best = None

        while quality >= min_quality:
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=quality, optimize=True)
            data = buf.getvalue()
            best = data
            if len(data) <= target_bytes:
                break
            quality -= 10

        if best:
            return best, "image/jpeg"
        return None
    except Exception as e:
        logger.warning(f"Image compression failed, using original: {e}")
        return None


def _extract_image_urls(text: str) -> List[str]:
    """Pick first few image URLs (Supabase public) from agent response.

    Note: WhatsApp via Twilio is most reliable with JPG/PNG media. We skip WEBP
    links to avoid silent delivery failures.
    """
    if not text:
        return []
    
    # First try to extract URLs from markdown image syntax: ![alt](url)
    markdown_images = re.findall(r'!\[.*?\]\((https?://[^)]+)\)', text)
    
    # Then fallback to plain URLs in text
    plain_urls = re.findall(r"https?://\S+", text)
    
    images: List[str] = []
    seen = set()
    
    # Prioritize markdown images (cleaner extraction)
    for url in markdown_images:
        clean_url = url.rstrip(').,;')
        if clean_url in seen:
            continue
        lower = clean_url.lower()
        if ("/storage/v1/object/" in lower) or lower.endswith(('.jpg', '.jpeg', '.png', '.webp')):
            # Drop WEBP for WhatsApp reliability
            if lower.endswith('.webp') or '.webp' in lower:
                continue
            images.append(clean_url)
            seen.add(clean_url)
        if len(images) >= MAX_MEDIA_PER_MESSAGE:
            break
    
    # Add plain URLs if still under limit
    if len(images) < MAX_MEDIA_PER_MESSAGE:
        for u in plain_urls:
            clean_url = u.rstrip(').,;')
            if clean_url in seen:
                continue
            lower = clean_url.lower()
            if ("/storage/v1/object/" in lower) or lower.endswith(('.jpg', '.jpeg', '.png', '.webp')):
                if lower.endswith('.webp') or '.webp' in lower:
                    continue
                images.append(clean_url)
                seen.add(clean_url)
            if len(images) >= MAX_MEDIA_PER_MESSAGE:
                break
    
    return images


def _normalize_whatsapp_number(raw: str) -> str:
    """Ensure Twilio WhatsApp numbers are prefixed exactly once."""
    if not raw:
        return ""
    return raw if raw.startswith("whatsapp:") else f"whatsapp:{raw}"


def _strip_media_from_text(text: str, media_urls: List[str]) -> str:
    """Remove markdown image tags and media URLs from body text."""
    if not text:
        return text

    cleaned = re.sub(r"!\[.*?\]\((https?://[^)]+)\)", "", text)
    if media_urls:
        for url in media_urls:
            cleaned = cleaned.replace(url, "")

    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


async def upload_to_supabase(path: str, content: bytes, content_type: str) -> bool:
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        logger.warning("Supabase credentials missing, cannot upload media")
        return False
    upload_url = f"{SUPABASE_URL}/storage/v1/object/{SUPABASE_STORAGE_BUCKET}/{path}"
    headers = {
        "Content-Type": content_type,
        "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
        "apikey": SUPABASE_SERVICE_KEY,
    }
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(upload_url, content=content, headers=headers)
        if resp.status_code in (200, 201):
            logger.info(f"✅ Uploaded media to Supabase: {path}")
            return True
        logger.warning(f"Supabase upload failed ({resp.status_code}): {resp.text}")
        return False
    except Exception as e:
        logger.error(f"Error uploading to Supabase: {e}")
        return False


async def analyze_image_with_vision(image_url: str) -> Optional[dict]:
    """Call OpenAI Vision API to analyze product image (same as webchat does)."""
    try:
        logger.info(f"🔍 Analyzing image with Vision API: {image_url[:80]}...")
        
        # Vision analysis prompts (matching webchat behavior)
        system_prompt = """Sen bir e-ticaret platformu için ürün görsellerini analiz eden bir asistansın.
        
Görevin:
1. Görseldeki ürünü tanımla (kategori/isim)
2. Ürünün durumunu değerlendir (yeni/kullanılmış/hasarlı)
3. Öne çıkan özellikleri listele
4. İçerik güvenliğini kontrol et (uygunsuz içerik varsa uyar)

JSON formatında cevap ver:
{
  "product": "ürün adı/kategorisi",
  "condition": "çok iyi görünüyor / kullanılmış / hasarlı",
  "features": ["özellik 1", "özellik 2", "özellik 3"],
  "safety_flags": []
}"""

        user_prompt = "Bu görseldeki ürünü analiz et ve yukarıdaki formatta JSON ile cevapla."
        
        # Make API call to Agent Backend's vision endpoint
        # We'll send it directly to OpenAI since we have access to OPENAI_API_KEY
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key:
            logger.error("❌ OPENAI_API_KEY environment variable not set - cannot analyze images. Check Railway Variables.")
            return None
        
        if not openai_key.startswith("sk-"):
            logger.error(f"❌ OPENAI_API_KEY format invalid (expected sk-***, got {openai_key[:10]}...)")
            return None
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {openai_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "gpt-4o-mini",
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": user_prompt},
                                    {"type": "image_url", "image_url": {"url": image_url}}
                                ]
                            }
                        ],
                        "max_tokens": 600,
                        "response_format": {"type": "json_object"}
                    }
                )
            except httpx.TimeoutException as te:
                logger.error(f"❌ Vision API timeout after 30s: {te}")
                return None
            except Exception as req_err:
                logger.error(f"❌ Vision API request failed: {req_err}")
                return None
        
        if not response.is_success:
            logger.error(f"❌ Vision API failed: status={response.status_code}")
            logger.error(f"Response text: {response.text[:500]}")
            return None
        
        try:
            result = response.json()
        except Exception as parse_err:
            logger.error(f"❌ Failed to parse Vision API response: {parse_err}")
            return None
        
        content = result.get("choices", [{}])[0].get("message", {}).get("content", "{}")
        
        try:
            analysis = json.loads(content)
            logger.info(f"✅ Vision analysis complete: {analysis.get('product', 'N/A')}")
            return analysis
        except json.JSONDecodeError as jde:
            logger.warning(f"⚠️ Vision API returned invalid JSON: {jde}")
            logger.warning(f"Content: {content[:200]}")
            return {"summary": content}
            
    except Exception as e:
        logger.error(f"❌ Vision analysis error: {e}")
        return None


async def process_media(
    user_id: str,
    listing_uuid: str,
    media_url: str,
    media_type: Optional[str],
    message_sid: Optional[str] = None,
    media_sid: Optional[str] = None,
) -> Optional[dict]:
    """Process media: download, compress, upload to Supabase, and analyze with Vision API.
    
    Returns:
        dict with 'path' and 'analysis', or None if processing failed
    """
    logger.info(f"🔄 Processing media: url={media_url[:80]}..., sid={message_sid}, media_sid={media_sid}")
    
    downloaded = await download_media(media_url, media_type, message_sid, media_sid)
    if not downloaded:
        logger.error(f"❌ Failed to download media from {media_url[:80]}")
        return None
    content, ctype = downloaded

    logger.info(f"🗜️ Compressing image ({len(content)} bytes)...")
    compressed = _compress_image(content, ctype)
    if compressed:
        content, ctype = compressed
        logger.info(f"✅ Compressed to {len(content)} bytes")

    storage_path = _build_storage_path(user_id, listing_uuid, ctype)
    logger.info(f"📤 Uploading to Supabase: {storage_path}")
    
    success = await upload_to_supabase(storage_path, content, ctype)
    if not success:
        logger.error(f"❌ Failed to upload media to Supabase")
        return None
    
    logger.info(f"✅ Media uploaded successfully: {storage_path}")
    
    # Now analyze the uploaded image with Vision API
    full_image_url = f"{SUPABASE_URL}/storage/v1/object/public/{SUPABASE_STORAGE_BUCKET}/{storage_path}"
    analysis = await analyze_image_with_vision(full_image_url)
    
    return {
        "path": storage_path,
        "analysis": analysis
    }


def get_conversation_history(phone_number: str) -> List[dict]:
    """Get conversation history for a phone number (Redis-first with fallback)"""
    # Try Redis first
    redis_key = f"conv:{phone_number}"
    redis_data = redis_client.get_json(redis_key)
    
    if redis_data and isinstance(redis_data, dict):
        messages = redis_data.get("messages", [])
        logger.info(f"📚 Retrieved {len(messages)} messages from Redis for {phone_number}")
        return messages
    
    # Fallback to in-memory
    if phone_number not in conversation_store:
        return []
    
    session = conversation_store[phone_number]
    if datetime.now() - session["last_activity"] > timedelta(minutes=CONVERSATION_TIMEOUT_MINUTES):
        logger.info(f"🕐 Conversation expired for {phone_number}, clearing history")
        del conversation_store[phone_number]
        return []
    
    return session["messages"]


def add_to_conversation_history(phone_number: str, role: str, content: str):
    """Add a message to conversation history (Redis-first with fallback)"""
    redis_key = f"conv:{phone_number}"
    
    # Get existing conversation from Redis
    redis_data = redis_client.get_json(redis_key)
    if not redis_data:
        redis_data = {"messages": [], "last_activity": datetime.now().isoformat()}
    
    # Add new message
    redis_data["messages"].append({
        "role": role,
        "content": content,
        "timestamp": datetime.now().isoformat()
    })
    redis_data["last_activity"] = datetime.now().isoformat()
    
    # Keep only last 20 messages
    if len(redis_data["messages"]) > 20:
        redis_data["messages"] = redis_data["messages"][-20:]
    
    # Save to Redis (30 min TTL)
    redis_client.set_json(redis_key, redis_data, ttl=1800)
    
    # Also update in-memory for fallback
    if phone_number not in conversation_store:
        conversation_store[phone_number] = {"messages": [], "last_activity": datetime.now()}
    
    conversation_store[phone_number]["messages"].append({
        "role": role,
        "content": content,
        "timestamp": datetime.now().isoformat()
    })
    conversation_store[phone_number]["last_activity"] = datetime.now()
    
    if len(conversation_store[phone_number]["messages"]) > 20:
        conversation_store[phone_number]["messages"] = conversation_store[phone_number]["messages"][-20:]
    
    logger.info(f"💾 Conversation updated (Redis + memory) for {phone_number}: {len(redis_data['messages'])} messages")


def update_search_cache(phone_number: str, results: List[dict]):
    """Store last search results (for detail requests) in memory."""
    if phone_number not in conversation_store:
        conversation_store[phone_number] = {"messages": [], "last_activity": datetime.now()}
    conversation_store[phone_number]["search_cache"] = {
        "results": results,
        "timestamp": datetime.now().isoformat()
    }
    conversation_store[phone_number]["last_activity"] = datetime.now()
    logger.info(f"💾 Search cache stored for {phone_number}: {len(results)} results")


def get_search_cache(phone_number: str) -> Optional[List[dict]]:
    cache = conversation_store.get(phone_number, {}).get("search_cache")
    if not cache:
        return None
    return cache.get("results") if isinstance(cache.get("results"), list) else None


def parse_search_cache_block(text: str) -> tuple[str, Optional[List[dict]]]:
    """Strip [SEARCH_CACHE] JSON block from text and return remaining text and parsed results."""
    if not text:
        return text, None

    marker = "[SEARCH_CACHE]"
    idx = text.find(marker)
    if idx == -1:
        return text, None

    stripped = text[:idx].rstrip()
    json_part = text[idx + len(marker):].strip()
    if not json_part:
        return stripped, None

    def _extract_balanced_json(raw: str) -> Optional[str]:
        first = raw[0]
        if first not in "[{":
            return None
        stack = [first]
        for i, ch in enumerate(raw[1:], start=1):
            if ch in "[{":
                stack.append(ch)
            elif ch in "]}":
                if not stack:
                    break
                last = stack.pop()
                if (last == "[" and ch != "]") or (last == "{" and ch != "}"):
                    return None
                if not stack:
                    return raw[: i + 1]
        return None

    raw_json = _extract_balanced_json(json_part) or json_part
    try:
        parsed = json.loads(raw_json)
    except Exception:
        try:
            parsed = ast.literal_eval(raw_json)
        except Exception as e:
            logger.warning(f"Failed to parse SEARCH_CACHE block: {e}")
            return stripped, None

    if isinstance(parsed, list):
        return stripped, parsed
    if isinstance(parsed, dict) and isinstance(parsed.get("results"), list):
        return stripped, parsed.get("results")
    return stripped, None


def build_last_search_results_note(results: List[dict], max_items: int = 10) -> str:
    """Build a compact note for backend to resolve "1 nolu ilan" requests.

    Format is aligned with backend expectations:
    [LAST_SEARCH_RESULTS] #1 id=... title=... | #2 id=... title=...
    """
    if not results:
        return ""
    
    # Emoji number mapping for better visibility (matching backend format)
    emoji_numbers = {1: "1️⃣", 2: "2️⃣", 3: "3️⃣", 4: "4️⃣", 5: "5️⃣", 6: "6️⃣", 7: "7️⃣", 8: "8️⃣", 9: "9️⃣", 10: "🔟"}
    
    parts: List[str] = []
    for i, item in enumerate(results[:max_items], start=1):
        if not isinstance(item, dict):
            continue
        listing_id = item.get("id")
        title = item.get("title") or ""
        if not listing_id:
            continue
        title_s = str(title).replace("|", " ").replace("\n", " ").strip()
        num_emoji = emoji_numbers.get(i, f"#{i}")
        parts.append(f"{num_emoji} id={listing_id} title={title_s}")
    if not parts:
        return ""
    return "[LAST_SEARCH_RESULTS] " + " | ".join(parts)


def clear_conversation_history(phone_number: str):
    """Clear conversation history for a phone number"""
    if phone_number in conversation_store:
        del conversation_store[phone_number]
        logger.info(f"🗑️ Conversation history cleared for {phone_number}")


def format_listing_detail(listing: dict) -> str:
    """Render a compact detail view for a single listing using cached search results."""
    title = listing.get("title") or "İlan"
    price = listing.get("price")
    location = listing.get("location") or "Belirtilmedi"
    condition = listing.get("condition") or "Belirtilmedi"
    category = listing.get("category") or "Belirtilmedi"
    listing_id = listing.get("id")
    owner_name = listing.get("user_name") or listing.get("owner_name")
    owner_phone = listing.get("user_phone") or listing.get("owner_phone")
    description = listing.get("description") or ""
    desc_short = description[:160] + ("..." if len(description or "") > 160 else "")

    lines = [title]
    if price is not None:
        lines.append(f"Fiyat: {price} TL")
    lines.append(f"Konum: {location}")
    lines.append(f"Durum: {condition}")
    lines.append(f"Kategori: {category}")
    if listing_id:
        lines.append(f"İlan ID: {listing_id}")
    if owner_name or owner_phone:
        parts = []
        if owner_name:
            parts.append(owner_name)
        if owner_phone:
            parts.append(owner_phone)
        lines.append("İlan sahibi: " + " | ".join(parts))
    if desc_short:
        lines.append(f"Açıklama: {desc_short}")

    signed_images = listing.get("signed_images") if isinstance(listing.get("signed_images"), list) else []
    imgs = signed_images[:3]
    if imgs:
        lines.append("Fotoğraflar:")
        lines.extend(imgs)
    else:
        lines.append("Fotoğraf yok")

    return "\n".join(lines)


def send_typing_indicator(phone_number: str, is_typing: bool) -> None:
    """
    Send typing indicator to WhatsApp user
    
    Args:
        phone_number: User's phone number (without whatsapp: prefix)
        is_typing: True to start typing, False to stop
    """
    if not twilio_client:
        logger.warning("⚠️ Twilio not configured, typing indicator not sent")
        return
    
    try:
        # Twilio doesn't have a direct "typing indicator" API for WhatsApp
        # However, we can send a read receipt which shows activity
        # Note: This is a best-effort feature; WhatsApp typing indicators are primarily client-side
        logger.info(f"{'✍️ Starting' if is_typing else '⏸️ Stopping'} typing indicator for {phone_number}")
        
        # Alternative approach: Send empty message to trigger "online" status
        # This is disabled by default as it may send unwanted messages
        # If you want to enable, uncomment below:
        # if is_typing:
        #     # This would send a silent notification, but may not work as expected
        #     pass
        
    except Exception as e:
        logger.warning(f"⚠️ Failed to send typing indicator: {e}")


def send_twilio_message(phone_number: str, body_text: str) -> None:
    """Send WhatsApp message via Twilio with length and media safeguards."""
    if not twilio_client:
        logger.warning("⚠️ Twilio not configured, response not sent")
        return

    media_urls = _extract_image_urls(body_text)
    
    # Log extracted URLs for debugging
    if media_urls:
        logger.info(f"📸 Extracted {len(media_urls)} media URLs:")
        for i, url in enumerate(media_urls):
            logger.info(f"  [{i+1}] {url}")

    # Twilio WhatsApp limit: body + media_url combined max 1600 chars
    MAX_WHATSAPP_LENGTH = 1600
    MEDIA_URL_OVERHEAD = len(media_urls) * 120  # Reserve 120 chars per media URL
    MAX_BODY_LENGTH = MAX_WHATSAPP_LENGTH - MEDIA_URL_OVERHEAD - 100  # Extra safety margin

    cleaned_body = _strip_media_from_text(body_text, media_urls)
    if not cleaned_body and media_urls:
        cleaned_body = "Sonuçlar görseller olarak gönderildi. Daha spesifik arama yapabilirsiniz."

    truncated_response = cleaned_body
    if len(cleaned_body) > MAX_BODY_LENGTH:
        logger.warning(
            f"⚠️ Response too long ({len(cleaned_body)} chars), truncating to {MAX_BODY_LENGTH} (with {len(media_urls)} media)"
        )
        truncated_response = cleaned_body[:MAX_BODY_LENGTH - 60] + "\n\n...(devamı için daha spesifik arama yapın)"

    try:
        message = twilio_client.messages.create(
            from_=_normalize_whatsapp_number(TWILIO_WHATSAPP_NUMBER),
            body=truncated_response,
            media_url=media_urls if media_urls else None,
            to=_normalize_whatsapp_number(phone_number)
        )
        logger.info(f"✅ Twilio message sent: {message.sid}")
    except Exception as e:
        logger.error(f"❌ Twilio send error: {e}")
        # Retry without media if media_urls caused the error
        if media_urls:
            logger.warning("🔄 Retrying without media...")
            try:
                message = twilio_client.messages.create(
                    from_=f'whatsapp:{TWILIO_WHATSAPP_NUMBER}',
                    body=truncated_response,
                    to=f'whatsapp:{phone_number}'
                )
                logger.info(f"✅ Message sent without media: {message.sid}")
            except Exception as retry_error:
                logger.error(f"❌ Retry also failed: {retry_error}")
                raise
        else:
            raise
# ================================================


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
    request: Request,
    Body: str = Form(""),  # Twilio bazen medya-only mesajlarda Body göndermiyor
    From: str = Form(...),
    To: str = Form(None),
    MessageSid: str = Form(None),
    NumMedia: int = Form(0),  # Number of media files (first index)
    MediaUrl0: str = Form(None),  # First media URL (kept for FastAPI schema)
    MediaContentType0: str = Form(None),  # First media content type (schema)
):
    """
    Twilio WhatsApp webhook endpoint
    
    Flow:
    1. Receive WhatsApp message from Twilio (text + optional media)
    2. Get conversation history for this phone number
    3. Send to Agent Backend (OpenAI Agents SDK with MCP tools)
    4. Store agent response in conversation history
    5. Send back via Twilio WhatsApp
    """
    form = await request.form()
    num_media = int(form.get("NumMedia", 0) or 0)
    media_keys = {k: v for k, v in form.items() if k.lower().startswith("media")}
    form_items = list(form.items())

    logger.info(f"📱 Incoming WhatsApp message from {From}: {Body}")
    logger.info(f"🔍 DEBUG - NumMedia: {num_media}, MediaUrl0: {MediaUrl0}, MediaContentType0: {MediaContentType0}")
    logger.info(f"🔍 DEBUG - MessageSid: {MessageSid}")
    logger.info(f"🧾 FORM MEDIA KEYS: {media_keys}")
    logger.info(f"🧾 FORM ITEMS (first 30): {form_items[:30]}")

    # Extract phone number early for history reuse
    phone_number = From.replace('whatsapp:', '')
    previous_history = get_conversation_history(phone_number)
    prev_draft_id, prev_media_paths = _extract_last_media_context(previous_history)

    # Check for media attachments (support multiple)
    media_items: List[tuple[str, Optional[str], Optional[str], Optional[str]]] = []
    for i in range(min(num_media, 10)):
        url = form.get(f"MediaUrl{i}")
        mtype = form.get(f"MediaContentType{i}")
        msid = form.get("MessageSid")
        # Twilio media SID is the last token of the URL
        media_sid = url.split("/")[-1] if url else None
        if url:
            media_items.append((url, mtype, msid, media_sid))

    if has_media := len(media_items) > 0:
        logger.info(f"🧾 MEDIA ITEMS PARSED: {media_items}")

    first_media_type = media_items[0][1] if has_media else None
    # Fresh media list for each new media message (vision needs only current photo)
    media_paths: List[str] = []
    vision_analyses: List[dict] = []  # Store vision analysis results
    draft_listing_id: Optional[str] = None

    if has_media:
        logger.info(f"📸 Media attached count: {len(media_items)}")
        draft_listing_id = str(uuid.uuid4())  # Always create new draft for vision analysis
        logger.info(f"📋 Draft listing ID: {draft_listing_id}")
        uploaded_any = False

        for idx, (url, mtype, msid, media_sid) in enumerate(media_items):
            logger.info(f"📸 Processing media {idx+1}/{len(media_items)}: {url[:80]}...")
            result = await process_media(phone_number, draft_listing_id, url, mtype, msid, media_sid)
            if result:
                uploaded_any = True
                media_paths.append(result["path"])
                if result.get("analysis"):
                    vision_analyses.append(result["analysis"])
                logger.info(f"✅ Media uploaded: {result['path']}")
                if result.get("analysis"):
                    logger.info(f"🔍 Vision analysis: {result['analysis'].get('product', 'N/A')}")
            else:
                logger.warning(f"Media upload failed for {url}")

        if uploaded_any and media_paths:
            # Build vision summary for conversation context
            vision_summary = ""
            for idx, analysis in enumerate(vision_analyses, 1):
                product = analysis.get("product", "")
                condition = analysis.get("condition", "")
                features = analysis.get("features", [])
                if product:
                    vision_summary += f"\n📷 Fotoğraf {idx}: {product}"
                if condition:
                    vision_summary += f" - {condition}"
                if features and isinstance(features, list):
                    vision_summary += f" | Özellikler: {', '.join(features[:3])}"
            
            # Add vision analysis to conversation history
            if vision_summary:
                vision_note = f"[VISION_ANALYSIS]{vision_summary}"
                add_to_conversation_history(phone_number, "assistant", vision_note)
            
            media_note = f"[SYSTEM_MEDIA_NOTE] DRAFT_LISTING_ID={draft_listing_id} | MEDIA_PATHS={media_paths}"
            add_to_conversation_history(phone_number, "assistant", media_note)
        else:
            logger.warning("Media processing failed; continuing without attachment")
            # Optional: notify user about media failure

    # Only send media paths/draft id when this request actually contained media
    payload_media_paths = media_paths if has_media else None
    payload_draft_id = draft_listing_id if has_media else None
    
    logger.info(f"📦 Sending to agent: draft_id={payload_draft_id}, media_count={len(payload_media_paths) if payload_media_paths else 0}")
    if payload_media_paths:
        logger.info(f"📸 Media paths being sent: {payload_media_paths}")

    # Optional legacy short-circuit: render detail from cached search results.
    # Default disabled to keep behavior consistent with WebChat and backend server-side detail rendering.
    lower_body = (Body or "").lower().strip()
    search_cache = get_search_cache(phone_number)
    detail_match = re.search(r"(\d+)\s*nolu\s*ilan[ıi]?\s*göster", lower_body)
    detail_idx: Optional[int] = None
    if detail_match:
        detail_idx = int(detail_match.group(1)) - 1
    elif search_cache and len(search_cache) == 1 and ("detay" in lower_body or "ilanı" in lower_body):
        detail_idx = 0

    if WHATSAPP_LOCAL_DETAIL_SHORTCIRCUIT and search_cache is not None and detail_idx is not None:
        if 0 <= detail_idx < len(search_cache):
            listing = search_cache[detail_idx]
            detail_text = format_listing_detail(listing)
            add_to_conversation_history(phone_number, "user", Body)
            add_to_conversation_history(phone_number, "assistant", detail_text)
            send_twilio_message(phone_number, detail_text)
            resp = MessagingResponse()
            return Response(content=str(resp), media_type="application/xml")
        else:
            warn_text = f"Bu aramada sadece {len(search_cache)} ilan var. 1-{len(search_cache)} arasından bir numara seçebilirsin."
            add_to_conversation_history(phone_number, "user", Body)
            add_to_conversation_history(phone_number, "assistant", warn_text)
            send_twilio_message(phone_number, warn_text)
            resp = MessagingResponse()
            return Response(content=str(resp), media_type="application/xml")

    try:
        user_message = Body
        
        # If message is empty but we have vision analysis, use it as the message
        if not user_message and vision_analyses:
            # Use the first media's product name as the search query
            first_analysis = vision_analyses[0]
            product_name = first_analysis.get("product", "")
            if product_name:
                user_message = product_name
                logger.info(f"🔍 Using vision analysis as message: {user_message}")

        # Get conversation history (previous messages only, NOT current message)
        conversation_history = get_conversation_history(phone_number)
        logger.info(f"📚 Conversation history for {phone_number}: {len(conversation_history)} messages")

        # If this is a numbered-detail request, inject last search results note so backend can resolve
        # listing id deterministically even across multiple backend instances.
        conversation_history_for_backend = list(conversation_history)
        if detail_idx is not None and isinstance(search_cache, list) and search_cache:
            note = build_last_search_results_note(search_cache)
            if note:
                conversation_history_for_backend.append({
                    "role": "assistant",
                    "content": note,
                    "timestamp": datetime.now().isoformat(),
                })
        
        # Show typing indicator before calling agent
        send_typing_indicator(phone_number, True)
        
        # Step 1: Call Edge Traffic Controller (PIN + 10min session gate) with conversation history + media
        # NOTE: current user_message is sent separately, NOT in history
        logger.info("🚦 Calling Edge Traffic Controller")
        agent_response = await call_agent_backend(
            user_message, 
            phone_number, 
            conversation_history_for_backend,
            media_paths=payload_media_paths,
            media_type=first_media_type if payload_media_paths else None,
            draft_listing_id=payload_draft_id
        )
        
        if not agent_response:
            raise HTTPException(status_code=500, detail="No response from Agent Backend")
        
        logger.info(f"✅ Agent response: {agent_response[:100]}...")

        # Extract and store search cache (if present), and strip it from user-facing text
        agent_response, search_cache_results = parse_search_cache_block(agent_response)
        if search_cache_results:
            update_search_cache(phone_number, search_cache_results)
            logger.info(f"📦 Search cache captured with {len(search_cache_results)} results")
        
        # Stop typing indicator before sending response
        send_typing_indicator(phone_number, False)
        
        # Step 2: Now add both user message and agent response to history
        add_to_conversation_history(phone_number, "user", user_message)
        add_to_conversation_history(phone_number, "assistant", agent_response)
        
        # Step 3: Send response back via Twilio WhatsApp
        send_twilio_message(phone_number, agent_response)
        
        # Return TwiML response (Twilio expects this)
        resp = MessagingResponse()
        return Response(content=str(resp), media_type="application/xml")
        
    except Exception as e:
        logger.error(f"❌ Error processing WhatsApp message: {str(e)}")
        logger.exception(e)
        
        # Stop typing indicator on error
        send_typing_indicator(phone_number, False)
        
        # Send error message to user
        resp = MessagingResponse()
        resp.message("Üzgünüm, bir hata oluştu. Lütfen daha sonra tekrar deneyin.")
        return Response(content=str(resp), media_type="application/xml")


async def call_agent_backend(
    user_input: str, 
    user_id: str, 
    conversation_history: List[dict],
    media_paths: Optional[List[str]] = None,
    media_type: Optional[str] = None,
    draft_listing_id: Optional[str] = None
) -> str:
    """
    Call Edge Function (Traffic Controller) → Backend
    
    Edge Function handles:
    - PIN verification (10-minute sessions)
    - Session timeout management
    - Rate limiting & security
    
    Args:
        user_input: User's message text
        user_id: User identifier (phone number)
        conversation_history: Previous messages in conversation
        media_paths: Optional list of uploaded storage paths
        media_type: Optional media content type (e.g., "image/jpeg")
        draft_listing_id: Optional UUID to keep storage paths and DB id aligned
        
    Returns:
        Agent's response text or PIN request message
    """
    if not EDGE_FUNCTION_URL or "YOUR_PROJECT.supabase.co" in EDGE_FUNCTION_URL or "YOUR_PROJECT" in EDGE_FUNCTION_URL:
        logger.error("EDGE_FUNCTION_URL not configured")
        return "Sistem yapılandırma hatası: WhatsApp güvenlik kapısı (EDGE_FUNCTION_URL) tanımlı değil."
    
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            # Call Edge Function (Traffic Police)
            logger.info(f"🚦 Calling Edge Function: {EDGE_FUNCTION_URL}")
            
            payload = {
                "source": "whatsapp",  # Important: identifies traffic source
                "phone": user_id,
                "message": user_input,
                "conversation_history": conversation_history,
                "media_paths": media_paths,
                "media_type": media_type,
                "draft_listing_id": draft_listing_id,
            }
            
            logger.info(f"📦 Payload: phone={user_id}, message_length={len(user_input)}, history_length={len(conversation_history)}")
            logger.info(f"🔍 DEBUG USER_ID FLOW: WhatsApp Bridge sending phone={user_id} to Edge Function")
            
            response = await client.post(
                EDGE_FUNCTION_URL,
                json=payload,
                headers={
                    "Content-Type": "application/json",
                    # Supabase Edge Functions: service role key for server-to-server calls
                    "Authorization": f"Bearer {SUPABASE_SERVICE_KEY or ''}",
                    "apikey": SUPABASE_SERVICE_KEY or "",
                }
            )
            
            result = response.json()
            logger.info(f"📨 Edge Function response: status={response.status_code}")
            logger.info(f"🔍 DEBUG USER_ID FLOW: Edge response contains user_id={result.get('user_id', 'NOT_PRESENT')}")
            
            # Check if PIN required
            if result.get("require_pin"):
                logger.info("🔒 PIN required - session expired or not exists")
                return result.get("response", "🔒 Güvenlik için PIN kodunuzu girin")
            
            # Check for errors
            if response.status_code == 403:
                logger.warning("⛔ Access denied - PIN required")
                return result.get("response", "⛔ Erişim reddedildi. PIN kodunuzu girin.")
            
            if response.status_code == 401:
                logger.warning("❌ Invalid PIN")
                return result.get("response", "❌ Hatalı PIN kodu")
            
            if not result.get("success"):
                logger.error(f"⚠️ Edge Function returned success=false")
                return result.get("response", "İşlem başarısız oldu. Lütfen tekrar deneyin.")
            
            response_text = result.get("response", "")
            if not response_text:
                logger.error("⚠️ Empty response from Edge Function")
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
            "twilio_configured": "yes" if twilio_client else "no",
            "active_conversations": len(conversation_store)
        }
    }


@app.post("/conversation/clear/{phone_number}")
async def clear_conversation(phone_number: str):
    """Clear conversation history for a phone number (admin endpoint)"""
    clear_conversation_history(phone_number)
    return {"status": "cleared", "phone_number": phone_number}


@app.get("/conversation/{phone_number}")
async def get_conversation(phone_number: str):
    """Get conversation history for a phone number (debug endpoint)"""
    history = get_conversation_history(phone_number)
    return {
        "phone_number": phone_number,
        "message_count": len(history),
        "messages": history
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
