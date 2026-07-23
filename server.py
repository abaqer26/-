import os
import json
import base64
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from anthropic import Anthropic

app = FastAPI(title="محلل الأكل بالصورة")

# ملاحظة: وقت الإطلاق الفعلي بدّل "*" بدومين موقعك/داشبوردك فقط
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

API_KEY = os.environ.get("ANTHROPIC_API_KEY")
MODEL = "claude-haiku-4-5-20251001"

# نبني عميل Anthropic بشكل كسول (مو وقت استيراد الملف) عشان لو صار أي خلل
# بالإعداد (مفتاح ناقص، مشكلة شبكة/بروكسي...) السيرفر يضل شغّال ويرجع خطأ
# واضح على الطلب نفسه، بدل ما يطيح كامل السيرفر وقت التشغيل.
_client = None
_client_error = None


def get_client():
    global _client, _client_error
    if _client is not None:
        return _client
    if not API_KEY:
        _client_error = "ما فيه ANTHROPIC_API_KEY مضبوط على السيرفر. حطه كمتغير بيئة قبل التشغيل."
        raise RuntimeError(_client_error)
    try:
        _client = Anthropic(api_key=API_KEY)
        return _client
    except Exception as e:
        _client_error = f"فشل تجهيز عميل Anthropic: {e}"
        raise RuntimeError(_client_error)

PROMPT = """أنت مساعد تغذية. انظر لصورة الأكل أو المنتج هذي وحدد:
1. اسم الأكلة أو المنتج بالعربي (وصف قصير وواضح)
2. حجم الحصة التقريبي
3. تقدير تقريبي لـ: السعرات، الكاربوهيدرات (غرام)، البروتين (غرام)، الدهون (غرام)، السكر (غرام لو موجود بوضوح وإلا 0)

رجّع ردك بصيغة JSON فقط، بدون أي نص أو شرح قبله أو بعده، بالضبط بهذا الشكل:
{"product_name": "...", "serving_note": "...", "calories": 0, "carbs_g": 0, "protein_g": 0, "fat_g": 0, "sugar_g": 0}

لو ما قدرت تتعرف على الأكل بوضوح من الصورة، رجّع "product_name": "غير واضح، صوّر بزاوية أوضح" مع باقي القيم صفر."""


TEXT_PROMPT_TEMPLATE = """أنت مساعد تغذية. الشخص وصف أكلته بالنص التالي: "{description}"

حدد:
1. اسم الأكلة بالعربي (نظّفه لو فيه أخطاء إملائية أو كان بالعامية)
2. حجم الحصة التقريبي (لو ما ذكره الشخص، افترض حصة عادية شائعة لهذي الأكلة)
3. تقدير تقريبي لـ: السعرات، الكاربوهيدرات (غرام)، البروتين (غرام)، الدهون (غرام)، السكر (غرام لو موجود بوضوح وإلا 0)

رجّع ردك بصيغة JSON فقط، بدون أي نص أو شرح قبله أو بعده، بالضبط بهذا الشكل:
{{"product_name": "...", "serving_note": "...", "calories": 0, "carbs_g": 0, "protein_g": 0, "fat_g": 0, "sugar_g": 0}}

لو الوصف غامض جدًا ومستحيل تقدّره، رجّع "product_name": "غير واضح، اكتب وصف أدق" مع باقي القيم صفر."""


class TextEstimateRequest(BaseModel):
    description: str


def parse_model_json(raw_text: str) -> dict:
    text = raw_text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    return json.loads(text)


@app.get("/health")
async def health():
    return {"status": "ok", "api_key_present": bool(API_KEY)}


@app.post("/analyze-food")
async def analyze_food(image: UploadFile = File(...)):
    try:
        client = get_client()
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    try:
        image_bytes = await image.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="ملف الصورة فاضي")

        b64_image = base64.b64encode(image_bytes).decode("utf-8")
        media_type = image.content_type or "image/jpeg"

        response = client.messages.create(
            model=MODEL,
            max_tokens=300,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": b64_image,
                            },
                        },
                        {"type": "text", "text": PROMPT},
                    ],
                }
            ],
        )

        raw_text = response.content[0].text
        data = parse_model_json(raw_text)
        return {"success": True, **data}

    except json.JSONDecodeError:
        raise HTTPException(
            status_code=502,
            detail="رد نموذج التحليل ما كان JSON صالح. جرب صورة ثانية أو صورة أوضح.",
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"خطأ غير متوقع: {e}")


@app.post("/estimate-food-text")
async def estimate_food_text(payload: TextEstimateRequest):
    try:
        client = get_client()
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    description = (payload.description or "").strip()
    if not description:
        raise HTTPException(status_code=400, detail="اكتب وصف الأكلة الأول")

    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=300,
            messages=[
                {
                    "role": "user",
                    "content": TEXT_PROMPT_TEMPLATE.format(description=description),
                }
            ],
        )

        raw_text = response.content[0].text
        data = parse_model_json(raw_text)
        return {"success": True, **data}

    except json.JSONDecodeError:
        raise HTTPException(
            status_code=502,
            detail="رد نموذج التحليل ما كان JSON صالح. جرب تكتب وصف أوضح.",
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"خطأ غير متوقع: {e}")
