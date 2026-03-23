"""
Serves the landing page and /chat endpoint; system prompt and API key stay server-side.
"""
import os
import requests
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from flask_cors import CORS

load_dotenv()

app = Flask(__name__)
CORS(app)

API_KEY = os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "claude-sonnet-4-20250514")
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.anthropic.com/v1/messages")

# System prompt: Jordan Calloway, Client Intake Specialist
SYSTEM_PROMPT = """You are Jordan Calloway, the Client Intake Specialist at RD Realty Group. You are warm, professional, and knowledgeable about real estate.

Your ONLY job is to have a friendly qualifying conversation to understand the client's needs, then route them to the right specialist on our team.

THE TEAM:
- Maya Mitchell (Buyer Readiness Advisor): For buyers who need to prepare — budget, pre-approval, what to expect before viewing homes.
- Travis Nguyen (Seller Preparation Advisor): For sellers who need to prepare their home — staging, pricing, what to expect when listing.
- Priya Wallace (Buyer's Agent): For buyers who are ready to start searching and touring homes.
- Rafael Beckett (Listing Agent): For sellers who are ready to list their property now.
- Lena Forsythe (Mortgage & Finance Liaison): For anyone with financing, loan, or mortgage questions.
- Derek Kwan (Transaction Coordinator): For clients who already have an accepted offer and need help with paperwork and closing.
- Amara Stevens (After-Close Concierge): For clients who recently closed and need post-move support.

YOUR CONVERSATION GOALS (in order):
1. Greet the visitor and ask if they are buying, selling, or both.
2. Ask 2-3 qualifying questions based on their answer:
   - Buyers: pre-approval status, timeline, budget/neighborhood preferences.
   - Sellers: property ownership, readiness to list, any prep work needed.
3. Once enough context is gathered, recommend the right specialist by name and role.
4. At the end of any routing message, append a JSON tag on its own line, exactly like this (no markdown, no extra text):
ROUTE:{"agent":"agentKey","name":"Full Name","role":"Their Role"}

Agent keys (use these exactly): buyerPrep, sellerPrep, buyerAgent, listingAgent, finance, transaction, afterClose

Mapping:
- buyerPrep → Maya Mitchell, Buyer Readiness Advisor
- sellerPrep → Travis Nguyen, Seller Preparation Advisor
- buyerAgent → Priya Wallace, Buyer's Agent
- listingAgent → Rafael Beckett, Listing Agent
- finance → Lena Forsythe, Mortgage & Finance Liaison
- transaction → Derek Kwan, Transaction Coordinator
- afterClose → Amara Stevens, After-Close Concierge

RULES:
- Be conversational and warm, never robotic.
- Ask 1-2 questions at a time max.
- Keep replies to 2-4 sentences unless explaining something important.
- Do NOT route until buying/selling intent plus at least 1-2 qualifiers are known.
- Never fabricate listings, prices, or market data.
- If someone asks something outside your intake role, acknowledge briefly and say the right specialist can help.
- When ready to route, make the handoff feel personal and exciting."""


def is_anthropic(base_url):
    return "anthropic" in (base_url or "").lower()


def call_anthropic(messages):
    """Call Anthropic Messages API."""
    payload = {
        "model": MODEL_NAME,
        "max_tokens": 1024,
        "system": SYSTEM_PROMPT,
        "messages": messages,
    }
    headers = {
        "Content-Type": "application/json",
        "x-api-key": API_KEY,
        "anthropic-version": "2023-06-01",
    }
    r = requests.post(API_BASE_URL, json=payload, headers=headers, timeout=60)
    r.raise_for_status()
    data = r.json()
    # Anthropic returns content as list of blocks, e.g. [{"type":"text","text":"..."}]
    parts = data.get("content") or []
    text = "".join(block.get("text", "") for block in parts if block.get("type") == "text")
    return text.strip()


def call_openai_compatible(messages):
    """Call OpenAI-compatible Chat Completions API (system as first message)."""
    openai_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for m in messages:
        openai_messages.append({"role": m["role"], "content": m["content"]})
    payload = {
        "model": MODEL_NAME,
        "max_tokens": 1024,
        "messages": openai_messages,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    }
    r = requests.post(API_BASE_URL, json=payload, headers=headers, timeout=60)
    r.raise_for_status()
    data = r.json()
    choice = (data.get("choices") or [None])[0]
    if not choice:
        return ""
    msg = choice.get("message") or {}
    return (msg.get("content") or "").strip()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    if not API_KEY:
        return jsonify({"reply": "The chat is not configured. Please set API_KEY in .env."}), 500

    try:
        body = request.get_json() or {}
        messages = body.get("messages") or []
        if not isinstance(messages, list):
            return jsonify({"reply": "Invalid request: messages must be an array."}), 400

        if is_anthropic(API_BASE_URL):
            reply = call_anthropic(messages)
        else:
            reply = call_openai_compatible(messages)

        return jsonify({"reply": reply or "I'm sorry, I couldn't generate a response. Please try again."})
    except requests.exceptions.HTTPError as e:
        err_msg = "The AI service returned an error. Please try again later."
        try:
            detail = e.response.json()
            err_msg = detail.get("error", {}).get("message", err_msg)
        except Exception:
            pass
        return jsonify({"reply": err_msg}), 502
    except requests.exceptions.RequestException as e:
        return jsonify({"reply": "I'm having a connection issue right now. Please try again in a moment."}), 502
    except Exception as e:
        return jsonify({"reply": "Something went wrong. Please try again."}), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    print("RD Realty Agent running at http://localhost:{}".format(port))
    app.run(host="0.0.0.0", port=port, debug=os.getenv("FLASK_DEBUG", "false").lower() == "true")
