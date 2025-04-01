import os
from flask import Flask, request, abort
from dotenv import load_dotenv
from linebot.v3.messaging import MessagingApi, Configuration, ApiClient
from linebot.v3.webhooks import WebhookHandler
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.webhooks import MessageEvent, TextMessageContent
from linebot.v3.messaging.models import TextMessage, ReplyMessageRequest
from rag_searcher import RagSearcher
from openai import OpenAI

# 環境変数読み込み
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")

client = OpenAI(api_key=OPENAI_API_KEY)
searcher = RagSearcher()

# Flaskアプリ
app = Flask(__name__)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

configuration = Configuration(access_token=LINE_CHANNEL_ACCESS_TOKEN)
line_bot_api = MessagingApi(ApiClient(configuration))

# ユーザーの職種・場所 状態管理
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

@handler.add(MessageEvent)
def handle_message(event):
    if not isinstance(event.message, TextMessageContent):
        return

    user_id = event.source.user_id
    msg = event.message.text.strip()

    # 状態初期化
    if user_id not in user_states:
        user_states[user_id] = {"role": None, "location": None}

    # 職種・現在地の判定（部分一致OK）
    lowered = msg.lower()
    if "研修医" in lowered:
        user_states[user_id]["role"] = "研修医"
        reply_text = "現在どこにいますか？（院内 or 院外）"
    elif "看護師" in lowered:
        user_states[user_id]["role"] = "看護師"
        reply_text = "現在どこにいますか？（院内 or 院外）"
    elif "医師" in lowered:
        user_states[user_id]["role"] = "医師"
        reply_text = "現在どこにいますか？（院内 or 院外）"
    elif "院外" in lowered:
        user_states[user_id]["location"] = "院外"
        reply_text = "質問をどうぞ"
    elif "院内" in lowered:
        user_states[user_id]["location"] = "院内"
        reply_text = "質問をどうぞ"
    else:
        role = user_states[user_id]["role"]
        location = user_states[user_id]["location"]

        # まだ role/location が決まってないなら案内
        if not role or not location:
            reply_text = "職種と現在地を先に教えてください（例：研修医、院外）"
        else:
            # 検索→GPT回答
            top_chunks = searcher.search_filtered(msg, role=role, location=location, top_k=3)
            context = "\n---\n".join(top_chunks)
            try:
                reply_text = ask_chatgpt_with_context(context, msg)
            except Exception as e:
                reply_text = f"エラーが発生しました: {str(e)}"

    # LINE返信
    line_bot_api.reply_message(
        ReplyMessageRequest(
            reply_token=event.reply_token,
            messages=[TextMessage(text=reply_text)]
        )
    )

if __name__ == "__main__":
    app.run(port=5000)
