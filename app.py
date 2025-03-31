import os
from flask import Flask, request, abort
from dotenv import load_dotenv
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage,
    PostbackEvent, PostbackAction, TemplateSendMessage,
    ButtonsTemplate
)
from openai import OpenAI
from rag_searcher import RagSearcher

load_dotenv()

app = Flask(__name__)

line_bot_api = LineBotApi(os.getenv("LINE_CHANNEL_ACCESS_TOKEN"))
handler = WebhookHandler(os.getenv("LINE_CHANNEL_SECRET"))

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
searcher = RagSearcher()

# 🔒 ユーザーの状態を一時保存（メモリ内）
user_states = {}

# 🚨 選択肢
ROLES = ["各科医師", "研修医", "看護師", "事務", "技師"]
LOCATIONS = ["病院内", "病院外"]

# 🧠 ChatGPTに投げる関数
def ask_chatgpt_with_context(context, question, role=None, location=None):
    conditions = []
    if role:
        conditions.append(f"職種: {role}")
    if location:
        conditions.append(f"現在位置: {location}")
    condition_str = "\n".join(conditions) if conditions else "（条件なし）"

    messages = [
        {"role": "system", "content": "あなたは岡崎市民病院の災害対応マニュアルを熟知した医療支援AIです。以下の条件とマニュアル抜粋を元に、質問に正確・簡潔に日本語で答えてください。"},
        {"role": "system", "content": f"【状況条件】\n{condition_str}"},
        {"role": "system", "content": f"【災害マニュアル抜粋】\n{context}"},
        {"role": "user", "content": f"質問：{question}"}
    ]
    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()

# 🟢 Webhookの受け口
@app.route("/callback", methods=["POST"])
def callback():
    signature = request.headers.get("X-Line-Signature")
    body = request.get_data(as_text=True)

    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)

    return "OK"

# 📩 通常のメッセージ受信時
@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    user_id = event.source.user_id
    user_message = event.message.text

    # ユーザー状態が未登録なら、初期化して職種を聞く
    if user_id not in user_states:
        user_states[user_id] = {"role": None, "location": None}
        ask_user_role(event.reply_token)
        return

    state = user_states[user_id]
    if not state["role"]:
        ask_user_role(event.reply_token)
        return
    if not state["location"]:
        ask_user_location(event.reply_token)
        return

    # 質問処理
    top_chunks = searcher.search_with_routing(user_message, top_k=3)
    context = "\n---\n".join(top_chunks)

    try:
        reply = ask_chatgpt_with_context(
            context,
            user_message,
            role=state["role"],
            location=state["location"]
        )
    except Exception as e:
        reply = f"エラーが発生しました: {str(e)}"

    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=reply)
    )

# 🔁 Postback（ボタン選択）時
@handler.add(PostbackEvent)
def handle_postback(event):
    user_id = event.source.user_id
    data = event.postback.data

    if user_id not in user_states:
        user_states[user_id] = {"role": None, "location": None}

    if data.startswith("role:"):
        user_states[user_id]["role"] = data.split(":")[1]
        ask_user_location(event.reply_token)
    elif data.startswith("location:"):
        user_states[user_id]["location"] = data.split(":")[1]
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text="職種と現在位置を登録しました。質問してください！")
        )

# 📦 職種を聞くボタン送信
def ask_user_role(reply_token):
    actions = [PostbackAction(label=role, data=f"role:{role}") for role in ROLES]
    buttons_template = ButtonsTemplate(
        title="あなたの職種を選んでください",
        text="以下から選択してください：",
        actions=actions[:4]  # 最大4つしか出せんので
    )
    message = TemplateSendMessage(alt_text="職種を選んでください", template=buttons_template)
    line_bot_api.reply_message(reply_token, message)

# 📦 現在位置を聞くボタン送信
def ask_user_location(reply_token):
    actions = [PostbackAction(label=loc, data=f"location:{loc}") for loc in LOCATIONS]
    buttons_template = ButtonsTemplate(
        title="現在どこにいますか？",
        text="以下から選択してください：",
        actions=actions
    )
    message = TemplateSendMessage(alt_text="現在位置を選んでください", template=buttons_template)
    line_bot_api.reply_message(reply_token, message)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
