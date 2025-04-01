import os
from flask import Flask, request, abort
from dotenv import load_dotenv
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage
from rag_searcher import RagSearcher
from openai import OpenAI

load_dotenv()
app = Flask(__name__)

line_bot_api = LineBotApi(os.getenv("LINE_CHANNEL_ACCESS_TOKEN"))
handler = WebhookHandler(os.getenv("LINE_CHANNEL_SECRET"))
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

searcher = RagSearcher()
user_states = {}

def ask_chatgpt_with_context(context, question):
    messages = [
        {"role": "system", "content": "あなたは病院の災害対応アシスタントです。以下はマニュアル抜粋です。"},
        {"role": "system", "content": f"マニュアル:\n{context}"},
        {"role": "user", "content": question}
    ]
    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()

@app.route("/callback", methods=["POST"])
def callback():
    signature = request.headers.get("X-Line-Signature")
    body = request.get_data(as_text=True)

    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)

    return "OK"

@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    user_id = event.source.user_id
    msg = event.message.text.strip()

    if user_id not in user_states:
        user_states[user_id] = {"role": None, "location": None}

    lowered = msg.lower()
    if "研修医" in lowered:
        user_states[user_id]["role"] = "研修医"
        reply = "現在どこにいますか？（院内 or 院外）"
    elif "看護師" in lowered:
        user_states[user_id]["role"] = "看護師"
        reply = "現在どこにいますか？（院内 or 院外）"
    elif "医師" in lowered:
        user_states[user_id]["role"] = "医師"
        reply = "現在どこにいますか？（院内 or 院外）"
    elif "院外" in lowered:
        user_states[user_id]["location"] = "院外"
        reply = "質問をどうぞ"
    elif "院内" in lowered:
        user_states[user_id]["location"] = "院内"
        reply = "質問をどうぞ"
    else:
        role = user_states[user_id]["role"]
        location = user_states[user_id]["location"]

        if not role or not location:
            reply = "職種と現在地を先に教えてください（例：研修医、院外）"
        else:
            chunks = searcher.search_filtered(msg, role=role, location=location, top_k=3)
            context = "\n---\n".join(chunks)
            try:
                reply = ask_chatgpt_with_context(context, msg)
            except Exception as e:
                reply = f"エラーが発生しました: {str(e)}"

    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=reply)
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
