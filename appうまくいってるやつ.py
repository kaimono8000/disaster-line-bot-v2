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



def ask_chatgpt_with_context(context, question):
    messages = [
        {"role": "system", "content": "あなたは岡崎市民病院の災害対応マニュアルを熟知した医療支援AIです。マニュアルの抜粋をもとに、質問に対して正確・簡潔に日本語で答えてください。"},
        {"role": "system", "content": f"【災害マニュアル抜粋】\n{context}"},
        {"role": "user", "content": f"質問：{question}"}
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
    user_message = event.message.text
    top_chunks = searcher.search(user_message, top_k=3)

    print("🧩 🔍 選ばれたチャンク（上位3件）:")
    for i, chunk in enumerate(top_chunks):
        print(f"[{i+1}] {chunk[:300]}...\n---\n")  # 先頭300文字だけ表示

    context = "\n---\n".join(top_chunks)

    try:
        reply = ask_chatgpt_with_context(context, user_message)
    except Exception as e:
        reply = f"エラーが発生しました: {str(e)}"

    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=reply)
    )


    

if __name__ == "__main__":
    app.run(debug=True)

