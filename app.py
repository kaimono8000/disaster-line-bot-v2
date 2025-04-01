import os
from flask import Flask, request, abort
from dotenv import load_dotenv

from linebot.v3.webhooks import WebhookHandler, MessageEvent, TextMessageContent
from linebot.v3.messaging import Configuration, ApiClient, MessagingApi
from linebot.v3.messaging.models import (
    TextMessage,
    TextSendMessage,
    QuickReply,
    QuickReplyButton,
    MessageAction
)

from rag_searcher import RagSearcher
from openai import OpenAI





load_dotenv()

app = Flask(__name__)
configuration = Configuration(access_token=os.getenv("LINE_CHANNEL_ACCESS_TOKEN"))
line_bot_api = MessagingApi(ApiClient(configuration))
handler = WebhookHandler(os.getenv("LINE_CHANNEL_SECRET"))
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
searcher = RagSearcher()

user_states = {}

def ask_chatgpt_with_context(context, question):
    messages = [
        {"role": "system", "content": "あなたは病院の災害マニュアルを元に質問に答えるアシスタントです。できるだけ簡潔に、正確に答えてください。"},
        {"role": "system", "content": f"マニュアル抜粋:\n{context}"},
        {"role": "user", "content": question}
    ]
    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        temperature=0.2
    )
    return response.choices[0].message.content.strip()

@app.route("/callback", methods=["POST"])
def callback():
    signature = request.headers.get("X-Line-Signature")
    body = request.get_data(as_text=True)
    try:
        handler.handle(body, signature)
    except Exception:
        abort(400)
    return "OK"

def ask_location(event):
    msg = TextSendMessage(
        text="災害発生時、あなたは今どこにいますか？",
        quick_reply=QuickReply(items=[
            QuickReplyButton(action=MessageAction(label="院内", text="院内")),
            QuickReplyButton(action=MessageAction(label="院外", text="院外")),
        ])
    )
    line_bot_api.reply_message(event.reply_token, msg)

def ask_role(event):
    msg = TextSendMessage(
        text="あなたの職種を選んでください",
        quick_reply=QuickReply(items=[
            QuickReplyButton(action=MessageAction(label="医師", text="医師")),
            QuickReplyButton(action=MessageAction(label="看護師", text="看護師")),
            QuickReplyButton(action=MessageAction(label="研修医", text="研修医")),
            QuickReplyButton(action=MessageAction(label="放射線技師", text="放射線技師")),
            QuickReplyButton(action=MessageAction(label="NICU直", text="NICU直")),
        ])
    )
    line_bot_api.reply_message(event.reply_token, msg)

def ask_question(event):
    msg = TextSendMessage(
        text="知りたいことを選んでください",
        quick_reply=QuickReply(items=[
            QuickReplyButton(action=MessageAction(label="どこへ行けばいい？", text="どこへ行けばいい？")),
            QuickReplyButton(action=MessageAction(label="何をすればいい？", text="何をすればいい？")),
            QuickReplyButton(action=MessageAction(label="その他の質問を入力", text="その他")),
        ])
    )
    line_bot_api.reply_message(event.reply_token, msg)

@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    user_id = event.source.user_id
    text = event.message.text.strip()

    if user_id not in user_states:
        user_states[user_id] = {"location": None, "role": None}

    state = user_states[user_id]

    if state["location"] is None:
        if text in ["院内", "院外"]:
            state["location"] = text
            ask_role(event)
        else:
            ask_location(event)
        return

    if state["role"] is None:
        state["role"] = text
        ask_question(event)
        return

    # location と role がそろったので質問に回答する
    try:
        chunks = searcher.search_filtered(
            query=text,
            role=state["role"],
            location=state["location"],
            top_k=3
        )
        context = "\n---\n".join(chunks)
        reply = ask_chatgpt_with_context(context, text)
    except Exception as e:
        reply = f"エラーが発生しました: {str(e)}"

    line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
