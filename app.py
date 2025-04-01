import os
from flask import Flask, request, abort
from dotenv import load_dotenv
from linebot.v3.webhook import WebhookHandler
from linebot.v3.messaging import MessagingApi, Configuration, ApiClient
from linebot.v3.messaging.models import TextMessage, TextMessageContent
from linebot.v3.webhooks import MessageEvent
from rag_searcher import RagSearcher
from openai import OpenAI

# 環境変数ロード
load_dotenv()

# Flaskアプリ初期化
app = Flask(__name__)

# LINE API初期化
configuration = Configuration(access_token=os.getenv("LINE_CHANNEL_ACCESS_TOKEN"))
line_bot_api = MessagingApi(ApiClient(configuration))
handler = WebhookHandler(os.getenv("LINE_CHANNEL_SECRET"))

# OpenAIクライアント初期化
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# RAG検索器
searcher = RagSearcher()

# ユーザーごとの状態（メモリ保持）
user_states = {}

# ChatGPTで回答を生成
def ask_chatgpt_with_context(context, question):
    messages = [
        {"role": "system", "content": "あなたは病院の災害対応マニュアルに基づいて答えるアシスタントです。できるだけ簡潔に、正確に答えてください。"},
        {"role": "system", "content": f"マニュアルの抜粋:\n{context}"},
        {"role": "user", "content": question}
    ]
    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()

# Webhook受信エンドポイント
@app.route("/callback", methods=["POST"])
def callback():
    signature = request.headers.get("X-Line-Signature")
    body = request.get_data(as_text=True)

    try:
        handler.handle(body, signature)
    except Exception as e:
        abort(400)

    return "OK"

# LINEメッセージ受信時の処理
@handler.add(MessageEvent, message=TextMessageContent)
def handle_message(event):
    user_id = event.source.user_id
    msg = event.message.text.strip()

    # ユーザーの状態を初期化
    if user_id not in user_states:
        user_states[user_id] = {"role": None, "location": None, "ready": False}

    state = user_states[user_id]

    # 役割が未設定なら職種として受け取る
    if not state["role"]:
        state["role"] = msg
        reply = "現在どこにいますか？（院内 or 院外）"
        line_bot_api.reply_message(event.reply_token, TextMessage(text=reply))
        return

    # 現在地が未設定なら設定
    if not state["location"]:
        if msg in ["院内", "院外"]:
            state["location"] = msg
            state["ready"] = True
            reply = "質問をどうぞ"
        else:
            reply = "現在どこにいますか？（院内 or 院外）"
        line_bot_api.reply_message(event.reply_token, TextMessage(text=reply))
        return

    # 職種・現在地がそろってるなら検索して回答
    if state["ready"]:
        try:
            role = state["role"]
            location = state["location"]
            top_chunks = searcher.search_filtered(msg, role=role, location=location, top_k=3)
            context = "\n---\n".join(top_chunks)
            reply = ask_chatgpt_with_context(context, msg)
        except Exception as e:
            reply = f"エラーが発生しました: {str(e)}"

        line_bot_api.reply_message(event.reply_token, TextMessage(text=reply))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
