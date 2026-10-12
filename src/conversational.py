"""
Conversational Intent Router and Small-Talk Handler for RAG.

Handles common conversational inputs (greetings, gratitude, well-being, identity)
instantly without unnecessary vector searches or irrelevant document refusals.
"""

import re
from typing import Optional, Tuple


class ConversationalRouter:
    """Detects and routes conversational pleasantries vs document queries."""

    # Patterns for pure chitchat / small-talk (English & Hindi/Hinglish)
    GREETING_PATTERNS = [
        r"^(hi|hello|hey|greetings|howdy|good\s+(morning|afternoon|evening|day))(\s+there|\s+bot|\s+ai)?[\s!.,?]*$",
        r"^(what\'?s\s+up|sup|yo)[\s!.,?]*$",
        r"^(नमस्ते|नमस्कार|प्रणाम|हेलो|राम\s*राम|namaste|namaskar|kya\s+haal\s+hai)[\s!.,?]*$",
    ]

    GRATITUDE_PATTERNS = [
        r"^(thanks?|thank\s+you|thx|ty|much\s+appreciated|appreciate\s+it)(\s+a\s+lot|\s+so\s+much|\s+very\s+much)?[\s!.,?]*$",
        r"^(great|awesome|perfect|good|cool|nice)(\s+thanks?|\s+thank\s+you)?[\s!.,?]*$",
        r"^(धन्यवाद|शुक्रिया|बहुत\s+धन्यवाद|बहुत\s+शुक्रिया|dhanyavaad|shukriya|bahut\s+shukriya)[\s!.,?]*$",
    ]

    WELLBEING_PATTERNS = [
        r"^(how\s+are\s+you|how\s+is\s+it\s+going|how\'?s\s+it\s+going|how\s+do\s+you\s+do|how\s+are\s+things|hope\s+you\s+are\s+well)[\s!.,?]*$",
        r"^(आप\s+कैसे\s+हैं|कैसे\s+हो|क्या\s+हाल\s+है|सब\s+कैसा\s+है|kaise\s+ho|aap\s+kaise\s+ho)[\s!.,?]*$",
    ]

    IDENTITY_PATTERNS = [
        r"^(who\s+are\s+you|what\s+are\s+you|what\s+can\s+you\s+do|what\s+is\s+your\s+purpose|help|what\s+do\s+you\s+do)[\s!.,?]*$",
        r"^(आप\s+कौन\s+हैं|तुम\s+कौन\s+हो|आप\s+क्या\s+कर\s+सकते\s+हैं|tum\s+kaun\s+ho|aap\s+kaun\s+ho)[\s!.,?]*$",
    ]

    FAREWELL_PATTERNS = [
        r"^(bye|goodbye|see\s+you|cya|take\s+care|have\s+a\s+(good|nice|great)\s+(day|night|one))[\s!.,?]*$",
        r"^(अलविदा|बाय|फिर\s+मिलेंगे|alvida|phir\s+milenge)[\s!.,?]*$",
    ]

    # Prefix patterns to strip from hybrid queries (e.g., "Hi, what is the stipend?", "नमस्ते, कार्य अनुभव क्या है?")
    GREETING_PREFIX_REGEX = re.compile(
        r"^(hi|hello|hey|good\s+(morning|afternoon|evening)|greetings|नमस्ते|नमस्कार|प्रणाम|हेलो|namaste)[\s,!.-]+",
        re.IGNORECASE,
    )

    @classmethod
    def classify_and_respond(cls, text: str) -> Tuple[bool, Optional[str], str]:
        """
        Classifies input text.
        Returns:
            (is_conversational: bool, canned_response: Optional[str], cleaned_query: str)
        """
        clean = text.strip()
        lower = clean.lower().strip()
        is_hindi = bool(re.search(r"[\u0900-\u097F]", clean))

        # 1. Greetings
        for pat in cls.GREETING_PATTERNS:
            if re.match(pat, lower, re.IGNORECASE):
                if is_hindi:
                    msg = (
                        "नमस्ते! 👋 मैं आपका दस्तावेज़ सहायक (Document Assistant) हूँ। "
                        "मैं आपके अपलोड किए गए दस्तावेज़ों से सत्यापित उत्तर और उद्धरण (citations) प्रदान कर सकता हूँ। "
                        "आज मैं आपकी क्या सहायता कर सकता हूँ?"
                    )
                else:
                    msg = (
                        "Hello! 👋 I am your enterprise Document Assistant. "
                        "I can search through your indexed documents and provide verified, grounded answers with citations. "
                        "How can I help you today?"
                    )
                return True, msg, clean

        # 2. Gratitude
        for pat in cls.GRATITUDE_PATTERNS:
            if re.match(pat, lower, re.IGNORECASE):
                if is_hindi:
                    msg = "आपका बहुत-बहुत स्वागत है! 😊 यदि आपके दस्तावेज़ों या नीतियों के संबंध में कोई अन्य प्रश्न हैं, तो अवश्य पूछें।"
                else:
                    msg = "You're very welcome! 😊 Feel free to ask if you have any more questions about your documents or policies."
                return True, msg, clean

        # 3. Well-being
        for pat in cls.WELLBEING_PATTERNS:
            if re.match(pat, lower, re.IGNORECASE):
                if is_hindi:
                    msg = "मैं बिल्कुल ठीक हूँ, पूछने के लिए धन्यवाद! 🚀 मैं आपके दस्तावेज़ों के विश्लेषण और प्रश्नों के उत्तर देने के लिए तैयार हूँ। आप क्या जानना चाहते हैं?"
                else:
                    msg = "I'm doing well, thank you for asking! 🚀 I'm ready to help you analyze and query your documents. What would you like to explore?"
                return True, msg, clean

        # 4. Identity & Help
        for pat in cls.IDENTITY_PATTERNS:
            if re.match(pat, lower, re.IGNORECASE):
                if is_hindi:
                    msg = (
                        "मैं एक एंटरप्राइज AI डॉक्यूमेंट Q&A असिस्टेंट हूँ। "
                        "मैं आपके अपलोड किए गए PDF और टेक्स्ट फाइलों को समझकर सटीक उत्तर और संदर्भ (citations) देता हूँ। "
                        "आप अनुभव, नीतियों या तकनीकी विवरणों के बारे में पूछ सकते हैं!"
                    )
                else:
                    msg = (
                        "I am an enterprise AI Document Q&A Assistant powered by hybrid retrieval and LLM intelligence. "
                        "I can search your uploaded PDFs, Markdown, and text files to answer questions with verifiable citations. "
                        "Try asking about equipment, policies, or on-call procedures!"
                    )
                return True, msg, clean

        # 5. Farewells
        for pat in cls.FAREWELL_PATTERNS:
            if re.match(pat, lower, re.IGNORECASE):
                if is_hindi:
                    msg = "अलविदा! आपका दिन शुभ हो, और दस्तावेज़ संबंधी किसी भी जानकारी के लिए कभी भी वापस आएं! 👋"
                else:
                    msg = "Goodbye! Have a great day, and feel free to return whenever you need documentation insights! 👋"
                return True, msg, clean

        # 6. Check for greeting prefix on an actual query (e.g. "Hi, what are the laptop options?", "नमस्ते, कार्य अनुभव क्या है?")
        stripped_query = cls.GREETING_PREFIX_REGEX.sub("", clean).strip()
        if stripped_query and len(stripped_query) >= 3 and stripped_query.lower() != lower:
            return False, None, stripped_query

        return False, None, clean
