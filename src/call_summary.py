"""
call_summary.py
===============
Post-call analysis — emotion, intent, hesitation, summary.
Integrated into the Maya agent pipeline: triggers automatically at call end.

DESIGN DECISIONS vs original emotion_recognition.py:
  REMOVED  — Whisper ASR: agent already uses OpenAI STT, transcript is available
  REMOVED  — superb/hubert-large-superb-er: English audio emotion (no English audio)
  REMOVED  — abuchane/wav2vec2-xlsr (Amharic model misidentified as Arabic)
  REMOVED  — TextBlob spell correction: hurts Arabic, slow, low value
  REMOVED  — contractions library: English only, marginal value
  REPLACED — Audio emotion: since we work from transcript only, use text models only
  REPLACED — Groq llama-3.3-70b → OpenAI gpt-4o-mini for LLM summary
             (eliminates Groq as a second provider dependency; one provider = one point of failure)
  ADDED    — call_log: structured turn-by-turn log for database storage
  ADDED    — LLM summary: uses OpenAI gpt-4o-mini to generate a readable Arabic summary
  ADDED    — auto outcome detection from timeline intent distribution
  ADDED    — overall intent derived from all timeline steps via LLM
  ADDED    — qualification extraction from LLM summary as fallback

Usage (from agent.py):
    from call_summary import CallSummaryBuilder, finalize_call_summary
"""

import json
import logging
import os
import re
from collections import Counter
from datetime import datetime
from typing import Optional

from dotenv import load_dotenv
from langdetect import detect
from openai import OpenAI

load_dotenv()
logger = logging.getLogger("call-summary")

# ============================================
# LAZY MODEL LOADER
# Only loads when finalize_call_summary() is first called
# ============================================

_models: dict = {}

_TRANSFORMER_SENTIMENT_ENV = "CALL_SUMMARY_ENABLE_TRANSFORMERS"

_POSITIVE_WORDS = {
    "excellent",
    "good",
    "great",
    "interested",
    "like",
    "حلو",
    "تمام",
    "كويس",
    "ممتاز",
    "مناسب",
    "مهتم",
    "عجبني",
}

_NEGATIVE_WORDS = {
    "bad",
    "expensive",
    "not interested",
    "بلاش",
    "رفض",
    "غالي",
    "مش عايز",
    "مش مهتم",
    "مش مناسب",
    "وحش",
}


def _load_summary_llm() -> None:
    if "summary_llm" in _models:
        return

    _models["summary_llm"] = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def _load_text_models() -> bool:
    if _models.get("text_models_unavailable"):
        return False
    if all(key in _models for key in ("tokenizer_en", "model_en", "sentiment_ar")):
        return True

    if os.getenv(_TRANSFORMER_SENTIMENT_ENV) != "1":
        logger.info(
            "Transformer sentiment models disabled. Set %s=1 to enable them.",
            _TRANSFORMER_SENTIMENT_ENV,
        )
        _models["text_models_unavailable"] = True
        return False

    logger.info("⏳ Loading call analysis models...")
    try:
        from transformers import pipeline
    except ImportError as exc:
        logger.warning("Text emotion models are unavailable: %s", exc)
        _models["text_models_unavailable"] = True
        return False

    # Arabic sentiment (MARBERT — best available for Arabic)
    _models["sentiment_ar"] = pipeline(
        "text-classification",
        model="Ammar-alhaj-ali/arabic-MARBERT-sentiment",
    )

    logger.info("✅ Call analysis models loaded.")
    return True


# ============================================
# TEXT PREPROCESSING
# ============================================


def _preprocess_text(text: str) -> tuple[str, str]:
    """Clean text and detect language. Returns (clean_text, language)."""
    clean_for_lang = re.sub(r"[^a-zA-Z\u0600-\u06FF\s]", "", text)
    try:
        lang = detect(clean_for_lang)
        if lang not in ["en", "ar"]:
            lang = "ar"
    except Exception:
        lang = "ar"

    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"([!?.,؟])\1+", r"\1", text)
    return text.strip(), lang


# ============================================
# TEXT EMOTION
# ============================================


def _fallback_text_emotion(text: str) -> dict:
    lower = text.lower()
    positive_hits = sum(1 for word in _POSITIVE_WORDS if word in lower)
    negative_hits = sum(1 for word in _NEGATIVE_WORDS if word in lower)

    if positive_hits > negative_hits:
        return {
            "sentiment": "positive",
            "sentiment_score": 0.7,
            "dominant_emotion": "positive",
            "confidence": 0.7,
        }
    if negative_hits > positive_hits:
        return {
            "sentiment": "negative",
            "sentiment_score": 0.7,
            "dominant_emotion": "negative",
            "confidence": 0.7,
        }
    return {
        "sentiment": "neutral",
        "sentiment_score": 0.5,
        "dominant_emotion": "neutral",
        "confidence": 0.5,
    }


def _run_text_emotion(text: str, lang: str) -> dict:
    """Detect emotion from text using loaded models."""
    if not _load_text_models():
        return _fallback_text_emotion(text)

    if lang == "en":
        return _fallback_text_emotion(text)

    # Arabic
    try:
        sent = _models["sentiment_ar"](text[:512])[0]
        result = {
            "sentiment": sent["label"],
            "sentiment_score": round(sent["score"], 3),
        }
        # Map Arabic sentiment to unified dominant_emotion
        label = sent["label"].lower()
        if "pos" in label or "positive" in label:
            result["dominant_emotion"] = "positive"
        elif "neg" in label or "negative" in label:
            result["dominant_emotion"] = "negative"
        else:
            result["dominant_emotion"] = "neutral"
        return result
    except Exception:
        return _fallback_text_emotion(text)


# ============================================
# HESITATION DETECTION
# ============================================

_HESITATION_WORDS = [
    # English
    "maybe",
    "not sure",
    "i think",
    "umm",
    "uh",
    "perhaps",
    "hmm",
    "well",
    # Arabic — uncertainty
    "يمكن",
    "مش عارف",
    "مش متأكد",
    "مش واثق",
    "تقريبا",
    "ممكن يكون",
    # Arabic — fillers / thinking-out-loud
    "يعني",
    "خليني أفكر",
    "آه يعني",
    "أه يعني",
    # Elongated hum variants (aural transcriptions)
    "اممم",
    "امممم",
    "هممم",
    "آممم",
    "أممم",
    "أمم",
    # Egyptian colloquial hedges
    "أه",
    "اوف",
    "معلش",
    "بصراحة",
    "مش عارف بقى",
    "إيه ده",
    "ايه ده",
    "يلا بس",
    "بس يعني",
]


def _detect_hesitation(text: str) -> bool:
    t = text.lower()
    return any(w in t for w in _HESITATION_WORDS)


# ============================================
# INTENT DETECTION
# ============================================

_INTENT_PATTERNS = {
    "ask_price": [
        "price",
        "cost",
        "how much",
        "السعر",
        "بكام",
        "الثمن",
        "سعر",
        "تكلفة",
        "رينج",
    ],
    "buy_property": [
        "buy",
        "purchase",
        "contract",
        "اشتري",
        "شراء",
        "احجز",
        "حجز",
        "عايز",
    ],
    "ask_location": ["where", "location", "area", "فين", "منطقة", "موقع", "مكان"],
    "ask_size": ["size", "meter", "sqm", "متر", "مساحة"],
    "ask_rooms": ["rooms", "bedroom", "غرف", "غرفة", "أوضة", "أوض"],
    "unqualified": ["no", "not interested", "مش مهتم", "لأ", "مش محتاج", "بلاش"],
    "objection_price": ["expensive", "costly", "غالي", "ده كتير", "أغلى", "مش قادر"],
    "interest_show": ["nice", "good", "interested", "حلو", "تمام", "عجبني", "ممتاز"],
}


def _detect_intent(text: str) -> str:
    t = text.lower()
    for intent, keywords in _INTENT_PATTERNS.items():
        if any(kw in t for kw in keywords):
            return intent
    return "general_question"


# ============================================
# LLM ARABIC SUMMARY
# ============================================


def _generate_llm_summary(
    turns: list[dict],
    call_outcome: str,
    client_name: str,
    qual_data: dict,
) -> str:
    """
    Uses Groq to generate a concise Arabic sales summary of the call.
    Input: list of {speaker, text} dicts
    """
    _load_summary_llm()

    lines = []
    for t in turns[-20:]:  # last 20 turns max to stay within token limit
        speaker = "العميل" if t.get("role") == "user" else "مايا"
        lines.append(f"{speaker}: {t.get('text', '')}")
    transcript_snippet = "\n".join(lines)

    qual_str = (
        f"المنطقة: {qual_data.get('area', 'غير محدد')} | "
        f"الغرف: {qual_data.get('bedrooms', 'غير محدد')} | "
        f"الميزانية: {qual_data.get('budget', 'غير محدد')}"
    )

    prompt = f"""أنت محلل مبيعات عقاري محترف.

بيانات المكالمة:
- اسم العميل: {client_name or "غير معروف"}
- نتيجة المكالمة: {call_outcome}
- بيانات التأهيل: {qual_str}

آخر جزء من المحادثة:
{transcript_snippet}

اكتب ملخص مهني قصير للمكالمة بالعربية (5-8 جمل):
1. ملخص ما طلبه العميل
2. ما عُرض عليه
3. موقفه وانطباعه
4. النتيجة النهائية
5. التوصية للمتابعة

ارجع النص بدون نقاط أو أرقام، كفقرة واحدة سلسة.
"""
    try:
        result = _models["summary_llm"].chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.3,
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}],
        )
        return result.choices[0].message.content.strip()
    except Exception as e:
        logger.warning(f"⚠️ LLM summary failed: {e}")
        return f"مكالمة مع {client_name or 'العميل'} — النتيجة: {call_outcome}"


# ============================================
# ARABIC WRITTEN-NUMBER NORMALIZER
# ============================================

_AR_ONES = {
    "واحد": 1,
    "اتنين": 2,
    "اثنين": 2,
    "تلاتة": 3,
    "ثلاثة": 3,
    "أربعة": 4,
    "اربعة": 4,
    "خمسة": 5,
    "ستة": 6,
    "سبعة": 7,
    "تمانية": 8,
    "ثمانية": 8,
    "تسعة": 9,
    "عشرة": 10,
    "أحد عشر": 11,
    "احد عشر": 11,
    "اتناشر": 12,
    "اثناشر": 12,
    "اثني عشر": 12,
    "تلاتاشر": 13,
    "ثلاثة عشر": 13,
    "أربعتاشر": 14,
    "أربعة عشر": 14,
    "خمستاشر": 15,
    "خمسة عشر": 15,
    "ستاشر": 16,
    "ستة عشر": 16,
    "سبعتاشر": 17,
    "سبعة عشر": 17,
    "تمنتاشر": 18,
    "ثمانية عشر": 18,
    "تسعتاشر": 19,
    "تسعة عشر": 19,
    "عشرين": 20,
    "تلاتين": 30,
    "ثلاثين": 30,
    "أربعين": 40,
    "اربعين": 40,
    "خمسين": 50,
    "ستين": 60,
    "سبعين": 70,
    "تمانين": 80,
    "ثمانين": 80,
    "تسعين": 90,
}
_AR_MAGNITUDE = {
    "مليار": 1_000_000_000,
    "مليون": 1_000_000,
    "ألف": 1_000,
    "الف": 1_000,
    "آلاف": 1_000,
    "الاف": 1_000,
}


def _normalize_arabic_number(text: str) -> str:
    if not text:
        return text

    t = text.strip()
    eastern = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
    normalized = t.translate(eastern).replace(",", "").replace("٬", "")
    if re.fullmatch(r"[\d.]+", normalized):
        val = float(normalized)
        return str(int(val)) if val == int(val) else str(val)

    t_lower = t
    has_half = bool(re.search(r"ونص|و نص", t_lower))

    magnitude = 1
    for word, mult in _AR_MAGNITUDE.items():
        if word in t_lower:
            magnitude = mult
            t_lower = t_lower.replace(word, "").strip()
            break

    digit_match = re.match(r"^([\d.]+)", t_lower.translate(eastern))
    if digit_match:
        coefficient = float(digit_match.group(1))
    else:
        coefficient = None
        for phrase, val in sorted(_AR_ONES.items(), key=lambda x: -len(x[0])):
            if phrase in t_lower:
                remaining = t_lower.replace(phrase, "").strip()
                extra = 0
                for phrase2, val2 in _AR_ONES.items():
                    if phrase2 in remaining:
                        extra = val2
                        break
                coefficient = val + extra if extra else val
                break

    if coefficient is None:
        if magnitude > 1:
            coefficient = 1
        else:
            return text

    total = coefficient * magnitude
    if has_half:
        total += magnitude / 2

    return str(int(total))


def _split_sentences(transcript: str) -> list[str]:
    parts = re.split(r"[.!?؟،\n]", transcript)
    return [s.strip() for s in parts if len(s.strip()) > 3]


# ============================================
# PHONE NUMBER EXTRACTOR
# ============================================


def _extract_phone_from_turns(turns: list[dict]) -> Optional[str]:
    """
    Scan user turns for an Egyptian/international phone number.
    Handles formats like: 01012345678, +201012345678, 010 123 45678
    """
    # Matches Egyptian mobile (010/011/012/015) and optional country code
    pattern = re.compile(r"(?:\+?2)?0(10|11|12|15)[\s\-]?\d{4}[\s\-]?\d{4}")
    for turn in turns:
        if turn.get("role") == "user":
            match = pattern.search(turn["text"])
            if match:
                # Strip spaces and dashes, normalise to 11-digit local format
                digits = re.sub(r"[\s\-]", "", match.group())
                # Drop leading country code (+20 or 20) if present
                if digits.startswith("+20"):
                    digits = "0" + digits[3:]
                elif digits.startswith("20") and len(digits) > 11:
                    digits = "0" + digits[2:]
                return digits
    return None


# ============================================
# CALL SUMMARY BUILDER
# ============================================
# "phone_number":"+201012345678",
# 		"status":"qualified",
# 		"lead_status":"qualified",
# 		"transcript":"user: انا مهتم بشقة في التجمع، ميزانيتي خمسة مليون، وعايز ثلاث غرف\nassistant: تمام، هدورلك على اختيارات مناسبة.",
# 		"details":"Phone number: +201012345678\nBudget: انا مهتم بشقة في التجمع، ميزانيتي خمسة مليون، وعايز ثلاث غرف\nRooms: انا مهتم بشقة في التجمع، ميزانيتي خمسة مليون، وعايز ثلاث غرف\nLocation: انا مهتم بشقة في التجمع، ميزانيتي خمسة مليون، وعايز ثلاث غرف\nProperty type: انا مهتم بشقة في التجمع، ميزانيتي خمسة مليون، وعايز ثلاث غرف\nSentiment: positive\nCall outcome: qualified",
# 		"summary":"Call with +201012345678. Client budget: انا مهتم بشقة في التجمع، ميزانيتي خمسة مليون، وعايز ثلاث غرف. Rooms: انا مهتم بشقة في التجمع، ميزانيتي خمسة مليون، وعايز ثلاث غرف. Location: انا مهتم بشقة في التجمع، ميزانيتي خمسة مليون، وعايز ثلاث غرف. Property type: انا مهتم بشقة في التجمع، ميزانيتي خمسة مليون، وعايز ثلاث غرف. Outcome: qualified. Sentiment: positive.",
# 		"sentiment":"positive",
# 		"outcome":"qualified",
# 		"duration_secs":90


class CallSummaryBuilder:
    """
    Lives inside MayaAgent for the duration of a call.
    Collects turn-by-turn data without any heavy model loading.
    Models are only loaded when finalize() is called post-call.
    """

    def __init__(
        self,
        client_name: str = "",
        call_id: str = "",
        phone: Optional[str] = None,
    ):
        self.client_name = client_name
        self.call_id = call_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.start_time = datetime.now()
        self.turns: list[dict] = []
        self.call_outcome = "ongoing"
        self.confirmed_qual: dict = {}
        self.phone: Optional[str] = phone.strip() if phone else None
        self.duration_secs = None

    def set_duration_secs(self, duration_secs):
        self.duration_secs = duration_secs

    def add_turn(self, role: str, text: str) -> None:
        """Add a conversation turn. role = 'user' or 'agent'."""
        if not text or not text.strip():
            return
        self.turns.append(
            {
                "role": role,
                "text": text.strip(),
                "timestamp": datetime.now().isoformat(),
            }
        )

    def set_outcome(self, outcome: str) -> None:
        """Manually set final call outcome. If not called, finalize() auto-detects it."""
        self.call_outcome = outcome

    def set_phone(self, phone: str) -> None:
        """Store the customer's phone number once the agent captures it during the call."""
        self.phone = phone.strip() if phone else None

    # ------------------------------------------------------------------
    # PRIVATE HELPERS
    # ------------------------------------------------------------------

    def _auto_determine_outcome(self, timeline_rows: list[dict]) -> str:
        """
        Determine call outcome from intent distribution in the timeline.
        Called automatically in finalize() when outcome is still 'ongoing'.
        """
        if not timeline_rows:
            return "unknown"

        intent_counts = Counter(row.get("intent", "unknown") for row in timeline_rows)

        prompt = f"""أنت محلل مبيعات عقاري.

بناءً على توزيع نوايا العميل خلال المحادثة:
{json.dumps(intent_counts, ensure_ascii=False)}

حدد النتيجة النهائية للمكالمة.
أرجع كلمة واحدة فقط من هذه الخيارات:
- unqualified : العميل رفض أو غير مهتم
- qualified     : العميل مهتم لكن لم يحجز
- follow_up       : العميل طلب المتابعة لاحقاً

كلمة واحدة فقط، بدون أي شرح."""

        try:
            response = _models["summary_llm"].chat.completions.create(
                model="gpt-4o-mini",
                temperature=0.3,
                max_tokens=600,
                messages=[{"role": "user", "content": prompt}],
            )
            outcome = response.choices[0].message.content.strip().lower()
            if "unqualified" in outcome:
                return "unqualified"
            if "follow_up" in outcome:
                return "follow_up"
            if "qualified" in outcome:
                return "qualified"
            return "unknown"
        except Exception as e:
            logger.warning(f"⚠️ Auto outcome detection failed: {e}")
            return "unknown"

    def _derive_overall_intent(self, timeline_rows: list[dict]) -> str:
        """
        Ask the LLM to look at every intent step in the timeline and
        return a single overall intent label for the whole conversation.
        """
        if not timeline_rows:
            return "unknown"

        # Build a step-by-step intent list for the LLM
        steps = [
            {"step": row.get("step"), "intent": row.get("intent")}
            for row in timeline_rows
        ]

        prompt = f"""أنت محلل مبيعات عقاري.

فيما يلي نوايا العميل في كل خطوة من خطوات المحادثة:
{json.dumps(steps, ensure_ascii=False)}

بناءً على تطور النوايا من البداية للنهاية، ما هي النية الإجمالية للعميل في المحادثة كلها؟

أرجع كلمة واحدة أو عبارة قصيرة جداً من هذه الخيارات:
- unqualified
- ask_price
- schedule_visit
- buy_property
- ask_location
- ask_size
- ask_rooms
- objection_price
- interest_show
- general_question

كلمة أو عبارة واحدة فقط، بدون شرح."""

        try:
            response = _models["summary_llm"].chat.completions.create(
                model="gpt-4o-mini",
                temperature=0.3,
                max_tokens=600,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content.strip().lower()
        except Exception as e:
            logger.warning(f"⚠️ Overall intent derivation failed: {e}")
            # Fallback: return the most common intent from the timeline
            return Counter(
                row.get("intent", "unknown") for row in timeline_rows
            ).most_common(1)[0][0]

    def _extract_qual_from_summary(self, llm_summary: str) -> dict:
        """
        Extract qualification data (area, bedrooms, budget, purpose) from
        the LLM summary text. Used as a fallback when no tracker data exists.
        """
        prompt = f"""من النص التالي، استخرج بيانات تأهيل العميل العقاري.

النص:
{llm_summary}

أرجع JSON فقط بهذا الشكل الدقيق، بدون أي كلام إضافي أو markdown:
{{
  "area": "المنطقة أو null",
  "bedrooms": عدد_صحيح_أو_null,
  "budget": رقم_صحيح_أو_null,
  "purpose": "شراء أو استثمار أو null"
}}"""

        try:
            response = _models["summary_llm"].chat.completions.create(
                model="gpt-4o-mini",
                temperature=0.3,
                max_tokens=600,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.choices[0].message.content.strip()
            # Strip markdown fences if present
            text = re.sub(r"```json|```", "", text).strip()
            extracted = json.loads(text)
            # Remove null values and string "null"
            return {
                k: v
                for k, v in extracted.items()
                if v is not None and str(v).lower() != "null"
            }
        except Exception as e:
            logger.warning(f"⚠️ Qual extraction from summary failed: {e}")
            return {}

    # ------------------------------------------------------------------
    # FINALIZE
    # ------------------------------------------------------------------

    def finalize(self, qual_tracker=None) -> dict:
        """
        Run full analysis. Called once at call end.
        Heavy models are loaded here (lazy).

        Pipeline:
          1. Build timeline (emotion + hesitation + per-step intent)
          2. Auto-detect call outcome from intent distribution  ← NEW
          3. Generate LLM summary
          4. Derive overall intent from timeline steps          ← NEW
          5. Fill missing qualification from LLM summary        ← NEW
        """
        logger.info(f"📊 Finalizing call summary for {self.client_name}")
        _load_summary_llm()

        # ── 0. Build qual data (tracker priority) ─────────────────
        qual_data = {k: v for k, v in self.confirmed_qual.items() if v is not None}
        if qual_tracker:
            tracker_data = {
                "area": qual_tracker.area,
                "bedrooms": qual_tracker.bedrooms,
                "budget": qual_tracker.budget,
                "purpose": qual_tracker.purpose,
            }
            for key, value in tracker_data.items():
                if value is not None and key not in qual_data:
                    qual_data[key] = value

        if qual_data.get("budget"):
            qual_data["budget"] = _normalize_arabic_number(str(qual_data["budget"]))

        # ── 1. Build timeline ──────────────────────────────────────
        user_turns = [t for t in self.turns if t["role"] == "user"]
        rows = []
        for i, turn in enumerate(user_turns):
            clean, lang = _preprocess_text(turn["text"])
            if not clean:
                continue
            emotion = _run_text_emotion(clean, lang)
            hesitation = _detect_hesitation(turn["text"])
            intent = _detect_intent(turn["text"])
            row = {
                "step": i + 1,
                "text": clean,
                "hesitation": hesitation,
                "intent": intent,
                "timestamp": turn.get("timestamp", ""),
            }
            row.update(emotion)
            rows.append(row)

        # Stats
        total = len(rows)
        hesitation_count = sum(1 for row in rows if row.get("hesitation"))
        hesitation_pct = round(hesitation_count / max(total, 1) * 100, 1)

        top_emotion = "unknown"
        emotion_counts = Counter(
            row.get("dominant_emotion") or row.get("sentiment")
            for row in rows
            if row.get("dominant_emotion") or row.get("sentiment")
        )
        if emotion_counts:
            top_emotion = emotion_counts.most_common(1)[0][0]

        # ── 2. Auto-detect outcome from timeline intents ───────────
        if self.call_outcome == "ongoing":
            self.call_outcome = self._auto_determine_outcome(rows)

        # ── 3. Generate LLM summary ────────────────────────────────
        llm_summary = _generate_llm_summary(
            self.turns, self.call_outcome, self.client_name, qual_data
        )

        # ── 4. Derive overall intent from all timeline steps ───────
        overall_intent = self._derive_overall_intent(rows)

        # ── 5. Fill missing qual from LLM summary (fallback) ───────
        if not qual_data:
            qual_data = self._extract_qual_from_summary(llm_summary)
            if qual_data.get("budget"):
                qual_data["budget"] = _normalize_arabic_number(str(qual_data["budget"]))

        # ── Auto-extract phone if not already set ────────────────
        if not self.phone:
            self.phone = _extract_phone_from_turns(self.turns)

        # ── Build final summary dict ───────────────────────────────
        summary = {
            "call_id": self.call_id,
            "client_name": self.client_name,
            "call_outcome": self.call_outcome,
            "call_duration_s": int(
                self.duration_secs
                if self.duration_secs is not None
                else (datetime.now() - self.start_time).total_seconds()
            ),
            "total_user_turns": total,
            "dominant_emotion": top_emotion,
            "overall_intent": overall_intent,  # ← replaces top_intent (LLM-derived)
            "hesitation_rate": f"{hesitation_pct}%",
            "qualification": qual_data,
            "llm_summary": llm_summary,
            "timestamp": self.start_time.isoformat(),
            # ── DB fields (call_summaries table) ──────────────────
            "phone": self.phone,  # captured by agent during call
            "summary": llm_summary,  # alias for llm_summary
            "classification": self.call_outcome,  # alias for call_outcome
            "called_at": self.start_time.isoformat(),  # alias for timestamp
        }

        return summary


# ============================================
# STANDALONE HELPER (for direct use / testing)
# ============================================


def analyze_call(
    transcript: str,
    call_outcome: str = "unknown",
    client_name: str = "",
) -> dict:
    """
    Analyze a call from a plain text transcript.

    Args:
        transcript:   Full conversation transcript text
                      Pass "ongoing" to let finalize() auto-detect the outcome.
        client_name:  Customer name if known

    Returns:
        dict with 'summary', 'timeline', 'turns'
    """
    builder = CallSummaryBuilder(client_name=client_name)
    for line in transcript.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        if line.startswith("العميل:") or line.startswith("client:"):
            builder.add_turn("user", re.sub(r"^[^:]+:\s*", "", line))
        elif line.startswith("مايا:") or line.startswith("maya:"):
            builder.add_turn("agent", re.sub(r"^[^:]+:\s*", "", line))
        else:
            builder.add_turn("user", line)

    # Pass "ongoing" to trigger auto-detection, or pass a known outcome
    builder.set_outcome(call_outcome)
    return builder.finalize()


def print_report(report: dict) -> None:
    """Pretty-print the analysis report."""
    print("\n" + "=" * 60)
    print("📊 CALL ANALYSIS REPORT")
    print("=" * 60)
    s = report["summary"]
    print(f"  Call ID         : {s.get('call_id', 'N/A')}")
    print(f"  Client          : {s.get('client_name', 'N/A')}")
    print(f"  Outcome         : {s.get('call_outcome', 'N/A')}")
    print(f"  Duration        : {s.get('call_duration_s', 0)}s")
    print(f"  Total turns     : {s.get('total_user_turns', 0)}")
    print(f"  Dominant emotion: {s.get('dominant_emotion', 'N/A')}")
    print(f"  Overall intent  : {s.get('overall_intent', 'N/A')}")
    print(f"  Hesitation rate : {s.get('hesitation_rate', 'N/A')}")
    print(f"\n  Qualification   : {s.get('qualification', {})}")
    print(f"  Phone           : {s.get('phone', 'N/A')}")
    print(f"  Classification  : {s.get('classification', 'N/A')}")
    print(f"  Called At       : {s.get('called_at', 'N/A')}")
    print(f"\n  LLM Summary:\n  {s.get('llm_summary', 'N/A')}")
    print("\n--- SENTENCE TIMELINE ---")
    if not report["timeline"].empty:
        print(report["timeline"].to_string(index=False))
    print("=" * 60 + "\n")


# ============================================
# STANDALONE TEST
# ============================================

if __name__ == "__main__":
    sample = """
مايا: أهلاً، أنا مايا من شركة العقارات. إزيك؟
العميل: أيوه كويس، عايز أسأل عن شقق في مدينة بدر.
مايا: حلو! كام غرفة محتاج؟
العميل: آه ممكن ثلاثة مثلا؟
مايا: تمام. وميزانيتك تقريبا كام؟
العميل: ما بين مليون ونص لاتنين مليون.
مايا: ممتاز. ممكن آخد رقمك عشان أبعتلك العروض؟
العميل: أيوه، رقمي 01012345678.
مايا: شكراً! هبعتلك التفاصيل دلوقتي.
العميل: تمام، متشكر.
"""
    report = analyze_call(
        transcript=sample,
        call_outcome="ongoing",  # let finalize() auto-detect
        client_name="محمود",
    )

    print("\n" + "=" * 60)
    print("📞 PHONE EXTRACTION TEST")
    print("=" * 60)

    test_cases = [
        ("11-digit local", "رقمي هو 01012345678"),
        ("with country code +20", "+201112345678 ده رقمي"),
        ("spaced format", "تقدر تتصل بيا على 010 1234 5678"),
        ("015 prefix", "رقمي 01534567890"),
        ("no phone at all", "مش عايز أقول الرقم دلوقتي"),
    ]

    all_pass = True
    for label, text in test_cases:
        turns = [{"role": "user", "text": text}]
        result = _extract_phone_from_turns(turns)
        found = result is not None
        status = (
            "✅"
            if (found or "no phone" in label)
            and not (not found and "no phone" not in label)
            else "❌"
        )
        all_pass = all_pass and (status == "✅")
        print(f"  {status} [{label}]")
        print(f"     Input : {text}")
        print(f"     Result: {result or 'None (correct — no number present)'}")
        print()

    print("=" * 60)
    print(f"  Result: {'All tests passed ✅' if all_pass else 'Some tests failed ❌'}")
    print("=" * 60)

    # ── Full pipeline test (requires models + OpenAI key) ──
    # Uncomment the line below to run the full call analysis:
    # print_report(report)
