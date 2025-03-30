import os
from dotenv import load_dotenv

print("カレントディレクトリ:", os.getcwd())
print("ファイル一覧:", os.listdir())

load_dotenv()

print("読み込んだLINE_CHANNEL_SECRET:", os.getenv("LINE_CHANNEL_SECRET"))

