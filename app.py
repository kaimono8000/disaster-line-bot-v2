# app.py
import os
from flask import Flask, request, abort
from dotenv import load_dotenv
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage
from openai import OpenAI
from rag_searcher import RagSearcher

load_dotenv()

app = Flask(__name__)
line_bot_api = LineBotApi(os.getenv("LINE_CHANNEL_ACCESS_TOKEN"))
handler = WebhookHandler(os.getenv("LINE_CHANNEL_SECRET"))
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
searcher = RagSearcher()

# ユーザー状態保持用（セッション管理簡易版）
user_states = {}

# ChatGPTへの問い合わせ（文脈つき）
import tiktoken

def truncate_tokens(text, max_tokens=6000, model="gpt-4"):
    enc = tiktoken.encoding_for_model(model)
    tokens = enc.encode(text)
    truncated = enc.decode(tokens[:max_tokens])
    return truncated

def ask_chatgpt_with_context(context, question):
    safe_context = truncate_tokens(context, max_tokens=6000)

    messages = [
        {"role": "system", "content": "以下は災害マニュアルの一部です。ユーザーの質問に対して、正確かつ簡潔に答えてください。"},
        {"role": "system", "content": f"マニュアル抜粋:\n{safe_context}"},
        {"role": "user", "content": question}
    ]

    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        temperature=0.2,
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

    # 職種を受け取る
    if msg in ["医師", "看護師", "研修医", "技師", "事務", "薬剤師"]:
        user_states[user_id]["role"] = msg
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage("今どこにいますか？（院内 or 院外）")
        )
        return

    # 現在地を受け取る
    if msg in ["院内", "院外"]:
        user_states[user_id]["location"] = msg
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage("わかりました。質問をどうぞ。")
        )
        return

    role = user_states[user_id].get("role")
    location = user_states[user_id].get("location")

    if not role or not location:
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage("職種と現在地を先に教えてください（例：研修医、院外）")
        )
        return

    # 条件付きベクトル検索
    try:
        top_chunks = searcher.search_filtered(msg, role=role, location=location, top_k=3)
        context = "\n---\n".join(top_chunks)
        reply = ask_chatgpt_with_context(context, msg)
    except Exception as e:
        reply = f"エラーが発生しました: {str(e)}"

    line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
