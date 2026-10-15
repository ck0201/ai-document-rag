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

    # Global System & FAQ Patterns (Default Cards and Operational Inquiries)
    GLOBAL_FORMATS_PATTERNS = [
        r"(what\s+(types?\s+of\s+)?(documents?|formats?|files?)|(document|file|upload)\s+formats?|(formats?|documents?|files?)\s+(are\s+)?supported|supported\s+formats?)",
        r"(what\s+(documents?|files?)\s+can\s+(i\s+)?upload|what\s+can\s+i\s+upload)",
        r"(file\s+size\s+limit|max(imum)?\s+file\s+size|storage\s+limit|how\s+big\s+(can|a)\s+file|upload\s+limits?|workspace\s+quota|storage\s+quota)",
        r"(how\s+(can\s+i|to|do\s+i)\s+upload|how\s+to\s+upload\s+(documents?|files?)|upload\s+a\s+document)",
        r"(कौन\s+से\s+दस्तावेज़|फाइल\s+फॉर्मेट|अपलोड\s+कैसे\s+करें|साइज\s+लिमिट)",
    ]

    GLOBAL_SECURITY_PATTERNS = [
        r"(protect\s+confidential\s+data|redact\s+pii|pii\s+masking|protect\s+data|data\s+privacy|security\s+(&|and)\s+privacy|how\s+secure|data\s+safe)",
        r"(prompt\s+injection|confidentiality|गोपनीयता|सुरक्षा|डेटा\s+सुरक्षित)",
    ]

    GLOBAL_HYBRID_RAG_PATTERNS = [
        r"(hybrid\s+retrieval|hybrid\s+search|reranking\s+work|dense\s+(&|and|\+)\s+sparse|how\s+does\s+retrieval\s+work|vector\s+search\s+\+\s+bm25|explain\s+the\s+rag\s+pipeline|hnsw)",
        r"(हाइब्रिड\s+सर्च|वेक्टर\s+सर्च)",
    ]

    GLOBAL_CAPABILITIES_PATTERNS = [
        r"(what\s+can\s+you\s+do\s+and\s+how\s+do\s+you\s+cite|how\s+do\s+you\s+cite\s+sources|how\s+do\s+citations\s+work|assistant\s+capabilities|cite\s+sources|what\s+are\s+your\s+features)",
        r"(उद्धरण|citations\s+कैसे|क्या\s+कर\s+सकते\s+हैं)",
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

        # 6. Global FAQ: Supported Document Formats & Upload Specifications
        for pat in cls.GLOBAL_FORMATS_PATTERNS:
            if re.search(pat, lower, re.IGNORECASE):
                if is_hindi:
                    msg = (
                        "### 📄 समर्थित दस्तावेज़ प्रारूप और अपलोड सीमाएँ\n\n"
                        "आप निम्नलिखित प्रारूपों में दस्तावेज़ अपलोड कर सकते हैं:\n"
                        "- **PDF दस्तावेज़ (`.pdf`)**: नीतियाँ, मैनुअल, तकनीकी विनिर्देश और रिपोर्ट।\n"
                        "- **Markdown फाइल्स (`.md`)**: तकनीकी दस्तावेज़, रीडमी और विकी गाइड।\n"
                        "- **Plain Text फाइल्स (`.txt`)**: नोट्स, सर्वर लॉग्स और टेक्स्ट दस्तावेज़।\n\n"
                        "**स्कूल नॉलेज बेस सीमाएँ और कोटा:**\n"
                        "- **अधिकतम फाइल आकार:** **50 MB** (वर्कस्पेस कोटा तक)।\n"
                        "- **कुल स्टोरेज सीमा:** **50 MB** प्रति स्कूल नॉलेज बेस।\n"
                        "- **अपलोड दर सीमा:** अधिकतम **5 अपलोड प्रति मिनट**।\n"
                        "- **इंडेक्सिंग:** फाइलें 150-कैरेक्टर ओवरलैप के साथ चंक की जाती हैं और PostgreSQL HNSW इंडेक्स में संग्रहीत होती हैं।\n\n"
                        "*दस्तावेज़ अपलोड और प्रबंधन केवल स्कूल एडमिन (पासकोड द्वारा अधिकृत) द्वारा किया जाता है। छात्र और अभिभावक किसी भी समय प्रश्न पूछ सकते हैं।*"
                    )
                else:
                    msg = (
                        "### 📄 Supported Document Formats & Upload Limits\n\n"
                        "You can upload documents in the following formats:\n"
                        "- **PDF Documents (`.pdf`)**: Circulars, fee structures, policy manuals, syllabi, and academic calendars.\n"
                        "- **Markdown Files (`.md`)**: Documentation, guides, schedules, and FAQs.\n"
                        "- **Plain Text Files (`.txt`)**: Notices, meeting transcripts, and text announcements.\n\n"
                        "**School Knowledge Base Quotas & Limits:**\n"
                        "- **Maximum File Size:** Up to **50 MB** (within school workspace quota).\n"
                        "- **Cumulative Storage Limit:** **50 MB** total storage across the school knowledge base.\n"
                        "- **Upload Rate Limit:** Up to **5 document uploads per minute**.\n"
                        "- **Fast Indexing:** Files are chunked recursively with 150-character semantic overlap and indexed in PostgreSQL using `pgvector` HNSW for sub-second retrieval.\n\n"
                        "*Document uploading and management is restricted to School Administration via Admin Passcode. Parents and students have instant, open access to ask questions anytime.*"
                    )
                return True, msg, clean

        # 7. Global FAQ: Security, Privacy & PII Protection
        for pat in cls.GLOBAL_SECURITY_PATTERNS:
            if re.search(pat, lower, re.IGNORECASE):
                if is_hindi:
                    msg = (
                        "### 🔒 सुरक्षा, गोपनीयता और PII संरक्षण\n\n"
                        "सिस्टम आपकी जानकारी को सुरक्षित रखने के लिए कई स्तरों पर सुरक्षा प्रदान करता है:\n"
                        "- **मल्टी-टेनेंट यूज़र आइसोलेशन:** प्रत्येक यूज़र को एक अलग वर्कस्पेस मिलता है। आपके दस्तावेज़ केवल आपके लिए निजी हैं।\n"
                        "- **स्वचालित PII रिडक्शन:** संवेदनशील व्यक्तिगत पहचान (आधार, पैन कार्ड, क्रेडिट कार्ड, एसएसएन, मोबाइल नंबर) स्वचालित रूप से मास्क हो जाती है।\n"
                        "- **प्रॉम्प्ट इंजेक्शन सुरक्षा:** सिस्टम प्रॉम्प्ट लीक और दुर्भावनापूर्ण निर्देशों को ब्लॉक करने के लिए गार्डरेल्स सक्रिय हैं।\n"
                        "- **DOMPurify सैनिटाइजेशन:** सभी आउटपुट को XSS हमलों से सुरक्षित किया जाता है।"
                    )
                else:
                    msg = (
                        "### 🔒 Security, Privacy & Confidentiality Architecture\n\n"
                        "The platform incorporates multi-layered enterprise defense for all documents and interactions:\n"
                        "- **Multi-Tenant User Isolation:** Every user receives an isolated workspace (`user_id`). Your uploaded documents, vector embeddings, and search queries are completely segregated—other users cannot view, query, or cite your files.\n"
                        "- **Automated PII Redaction:** Sensitive Personally Identifiable Information (such as Aadhaar numbers, PAN cards, credit cards, US SSNs, phone numbers, and email addresses) is automatically detected and masked (`[REDACTED_...]`) before prompts are built.\n"
                        "- **Prompt Injection Defense:** Input queries are screened with heuristic guardrails against jailbreak directives, delimiter overrides, and system prompt leakage attempts.\n"
                        "- **Stored XSS Sanitization:** All markdown responses pass through DOMPurify before rendering in the browser to neutralize any embedded scripts.\n"
                        "- **Rate Limiting:** Free-tier rate limiting (12 RPM for queries) prevents quota exhaustion and denial-of-service."
                    )
                return True, msg, clean

        # 8. Global FAQ: Hybrid Retrieval (Dense + Sparse) & Reranking
        for pat in cls.GLOBAL_HYBRID_RAG_PATTERNS:
            if re.search(pat, lower, re.IGNORECASE):
                if is_hindi:
                    msg = (
                        "### ⚡ हाइब्रिड रिट्रीवल (Dense + Sparse) और री-रैंकिंग\n\n"
                        "यह प्रणाली दोहरे चरण वाली आधुनिक रिट्रीवल पाइपलाइन का उपयोग करती है:\n"
                        "1. **Dense Vector Search:** FastEmbed (384-आयामी वेक्टर्स) और PostgreSQL `pgvector` HNSW कोसाइन समानता।\n"
                        "2. **Sparse BM25 Search:** सटीक कीवर्ड्स, कोड्स और तकनीकी शब्दों की खोज।\n"
                        "3. **Reciprocal Rank Fusion (RRF):** सघन और विरल दोनों परिणामों का संयोजन।\n"
                        "4. **Cross-Encoder Reranking:** शीर्ष चंक्स की प्रासंगिकता का पुनर्मूल्यांकन।\n"
                        "5. **सटीक ग्राउंडिंग:** उत्तर केवल सत्यापित स्रोतों और उद्धरणों के साथ दिए जाते हैं।"
                    )
                else:
                    msg = (
                        "### ⚡ Hybrid Retrieval (Dense + Sparse) & Reranking Architecture\n\n"
                        "The platform implements a production two-stage hybrid retrieval architecture:\n"
                        "1. **Dense Semantic Search (FastEmbed):** Transforms text into 384-dimensional dense vectors (BAAI/bge-small-en-v1.5) and uses PostgreSQL `pgvector` HNSW cosine distance (`<=>`) for semantic intent matching.\n"
                        "2. **Sparse BM25 Search:** Performs exact token matching to accurately capture product codes, acronyms, model numbers, and technical jargon.\n"
                        "3. **Reciprocal Rank Fusion (RRF):** Blends the dense and sparse candidate rankings into an optimal candidate set.\n"
                        "4. **Cross-Encoder Reranking:** A neural cross-encoder evaluates candidates directly against the query, filtering out irrelevant false positives.\n"
                        "5. **Strict Grounding:** The LLM generates responses strictly from the top reranked chunks with verifiable citation footnotes."
                    )
                return True, msg, clean

        # 9. Global FAQ: Assistant Capabilities & Citation Engine
        for pat in cls.GLOBAL_CAPABILITIES_PATTERNS:
            if re.search(pat, lower, re.IGNORECASE):
                if is_hindi:
                    msg = (
                        "### 💡 सहायक क्षमताएं और उद्धरण (Citations)\n\n"
                        "मैं आपके दस्तावेज़ों के लिए निम्नलिखित कार्य कर सकता हूँ:\n"
                        "- **सत्यापित उत्तर:** बिना किसी गलत सूचना (hallucination) के आपके दस्तावेज़ों के आधार पर सटीक उत्तर।\n"
                        "- **प्रत्यक्ष उद्धरण (Citations):** प्रत्येक उत्तर में स्रोत फाइल का नाम, पृष्ठ संख्या और समानता स्कोर शामिल होता है।\n"
                        "- **बहु-स्तरीय संवाद (Multi-Turn):** पिछले प्रश्नों और संदर्भ को याद रखना।\n"
                        "- **रीयल-टाइम स्ट्रीमिंग:** तत्काल टोकन स्ट्रीमिंग और लेटेंसी मेट्रिक्स।"
                    )
                else:
                    msg = (
                        "### 💡 Assistant Capabilities & Citation Engine\n\n"
                        "Here is what I can do for you:\n"
                        "- **Strictly Grounded Answers:** Answer questions directly from your indexed documents with zero unverified hallucinations.\n"
                        "- **Verifiable Source Footnotes:** Every answer provides explicit citations containing the source filename, page number, chunk ID, and cosine similarity score.\n"
                        "- **Multi-Turn Conversational Memory:** Understand follow-up questions and automatically reformulate pronouns and ambiguous references using session context.\n"
                        "- **Streaming Telemetry:** Deliver tokens via Server-Sent Events (SSE) with real-time latency breakdowns (retrieval ms vs. generation ms).\n"
                        "- **Workspace Privacy:** Ensure all your documents remain completely private to your personal workspace.\n\n"
                        "*Click on any citation badge below an answer to inspect the exact text excerpt and matching confidence score!*"
                    )
                return True, msg, clean

        # 6. Check for greeting prefix on an actual query (e.g. "Hi, what are the laptop options?", "नमस्ते, कार्य अनुभव क्या है?")
        stripped_query = cls.GREETING_PREFIX_REGEX.sub("", clean).strip()
        if stripped_query and len(stripped_query) >= 3 and stripped_query.lower() != lower:
            return False, None, stripped_query

        return False, None, clean
