#!/usr/bin/env python3

'''
dialog_v3.2


[概要]
    rinna/japanese-gpt-neox-3.6b-instruction-sft
    https://huggingface.co/rinna/japanese-gpt-neox-3.6b-instruction-sft

    対話型にファインチューニングされたrinna3.6B-instruction-sftを用いることで、
    CLI上でAIとチャットを出来るようにしたプログラムです。


[修正履歴]
  ・ タグを[reset]から[clear]に変更
  ・ トークナイザが選択出来るように修正
  ・ 新オプション[remem]の実装に伴い対話ループの順序変更
  ・ 環境構築メモの見直し



[失敗]

  ・ VRAM削減  -> "model under dtype=torch.uint8 since it is not a floating point dtype"
        https://megalodon.jp/2023-0520-0713-27/https://note.nkmk.me:443/python-pytorch-dtype-to/

  
[テスト環境]
    cuda vram使用率推移
        - RTX4090, vram24GB, max_length=256設定
            アイドル時 : 2.4 GB
            実行時 : 17.6 GB
            5発話でmax_lengthがフロー : 18.1 GB
            レスポンスタイム : 1 ~ 5秒 (体感)

        - TESLA P40, vram24GB, max_length=256設定
            アイドル時 : 0 GB
            実行時 : 15.7 GB
            5発話でmax_lengthがフロー : 16.3 GB
            レスポンスタイム : 5 ~ 8秒 (体感)
            備考 : ebayで中古3万円、お得!

        - RTX3080Ti, vram12GB, float16指定, max_length=256設定
            アイドル時 : 0 GB
            実行時 : 8.4 GB
            5発話でmax_lengthがフロー :  8.8 GB
            レスポンスタイム : 2 ~ 5秒 (体感)
            備考 : SATA SSDで実行。実行時の読み込みに60秒くらい掛かる模様
            
    cpu RAM使用率推移
        - 5950x, RAM64GB, max_length=256設定
            アイドル時 : 8.0 GB
            実行時 : 28.3 GB(ピーク) ～ 21.6 GB(安定)
            5発話でmax_lengthがフロー : 21.7 GB
            レスポンスタイム : 10 ~ 13秒 (体感)
            
    パッケージのversion
        - cuda 11.7
        - python == 3.10.6
        - torch == 1.13.1+cu117
        - transformers == 4.29.2
  
    
[環境構築メモ]

    <参考: 仮想環境: Python環境構築ガイド>
        https://www.python.jp/install/windows/venv.html

    ・コマンドプロンプトを管理者権限で実行

    D:\> git clone https://github.com/AlgosErgo/rinna_dialog
        
    D:\> cd rinna_dialog

    D:\rinna_dialog> py -3.10 -m venv rinna_venv

    D:\rinna_dialog> .\rinna_venv\Scripts\activate.bat

    (rinna_venv) D:\rinna_dialog> pip install -r requirements.txt

    (rinna_venv) D:\rinna_dialog> python.exe -m pip install --upgrade pip

    (rinna_venv) D:\rinna_dialog> pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
 
    (rinna_venv) D:\rinna_dialog> python dialog.py


    Tips
      ・起動するとHuggingFaceから自動でモデルがダウンロードされ以下のディレクトリへキャッシュされる。
          - C:\\Users\\{ユーザー名}\\.cache\\huggingface\\hub
      ・モデルを自前でダウンロードした場合は、後述の[Q&A]の手順でキャッシュ化が必要

[次回以降起動手順]

    D:\> cd rinna_dialog
         
    D:\rinna_dialog> .\rinna_venv\Scripts\activate.bat
         
    (rinna_venv) D:\rinna_dialog> python dialog.py

   Tips
     ・venvを使用せず「pathを通したpython」に直接要求パッケージをいれるとpython dialog.pyだけで動く。
          - ただし環境も混ざるので一長一短。
     ・バッチファイルとかで自動化すると便利。


[小ネタ]
    Voiceroidやsoftalkなどの音声合成ソフトを持っている人は「AssistantSeika」のSeikaSay2をsubprocesで使うとAIの返答を喋らせることができちゃうゾ
    まだ試してないけどvoicevoxなどのAPIでもワンチャン

[Q&A]
    Q : なんかエラーが出た！
    A : 環境構築メモで再構築してみましょう。それでも動かないときはLLM部まで

    Q : 起動時に"t = torch.tensor([], （略）"と出て固まるぞ！
    A : 圧縮に時間が掛かっているようです。
        タスクマネージャーを開いてRAMが使われていくのを眺めて気長に待ちましょう。
        VRAMへの読み込みが終わっているのに動かない場合はEnterキーを押すと強制的に進めます。

   Q : モデルのダウンロードが途中で止まってしまうんやが・・・
   A : ブラウザでダウンロードした「pytorch_model.bin」をキャッシュの形式にして読み込ませることができます。

       ・「rinna_3.6b_withoutmodel_pack.zip」を解凍のち以下へ移動
         -> C:\\Users\\{users}\\.cache\\huggingface\\hub

       ・ 中身を開いて以下のtxtを探す
          models--rinna--japanese-gpt-neox-3.6b-instruction-sft\blobs\0c6124c628f8ecc29be1b6ee0625670062340f5b99cfe543ccf049fa90e6207b.txt

       ・ pytorch_model.binを「0c6124c628f8ecc29be1b6ee0625670062340f5b99cfe543ccf049fa90e6207b」にリネーム

       ・ 最後にtxtファイルをゴミ箱へいれて完了

       <完成例>
        https://i.imgur.com/4Crp8qG.png
'''
##################################################################################################################################################




#################
###  設定項目  ###
#################


####
# 好きな名前を入れてね
#
user_name = "ユーザー"
ai_name = "AI"


####
# AIの短期記憶リスト(会話履歴)　お好みで書き換え可。
# 対話ループの都合上、AI応答 -> user入力の繰り返しなので、
# listの最後はuserとなるのが好ましい。（もちろんコード側を書き換えてもOK）
#
conversation_list = [
    {"speaker": user_name, "text": "このプログラムはなんですか？"},
    {"speaker": ai_name, "text": "りんなを即席対話型にしたものです"},
    {"speaker": user_name, "text": "あなたは誰ですか？"},
]


####
# モデルの移動先。
# "cuda"か"cpu"を指定。
#
processor = "cuda"


####
# モデルを圧縮するオプション
#   - True  : 圧縮    【vram使用率:約8GB】
#   - False : 未圧縮  【vram使用率:約16GB】
#
f16_mode = True


####
# max_lengthを増やすと会話が長続きする。
# ※ただしvramと要相談。
#
token_max_lengh = 1024


####
# max_lengthがあふれた場合の動作
# (未実装)
#
# auto_clear = False


####
# temperatureが低いほど一貫性がある出力を、
# temperatureが高くなるほど多様な出力をするようになる、らしい。
#　<参考 : GPTのtemperatureとは何なのか : https://megalodon.jp/2023-0519-0821-24/https://qiita.com:443/suzuki_sh/items/8e449d231bb2f09a510c>
#
token_temperature = 0.7


####
# トークナイザ名。
# "AutoTokenizer" or "T5Tokenizer"
#
tokenizer_name = "AutoTokenizer"


####
# モデル名。
#
model_name = "rinna/japanese-gpt-neox-3.6b-instruction-sft"



### ここからコード
###################################################################################

#import subprocess
import torch
from transformers import T5Tokenizer, AutoTokenizer, AutoModelForCausalLM, file_utils


### 事前学習済みモデルの読み込み
if f16_mode == True:
    model = AutoModelForCausalLM.from_pretrained(
                 model_name, torch_dtype=torch.float16
             )

elif f16_mode == False:
    model = AutoModelForCausalLM.from_pretrained(
                 model_name
             )

else:
    print("[Err] f16_modeの値が不正です。デフォルト設定のfloat16モードで実行します。")
    model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.float16
             )


### トークナイザの読み込み
if tokenizer_name == "AutoTokenizer":
    tokenizer = AutoTokenizer.from_pretrained(
                model_name, use_fast=False
             )
    
elif tokenizer_name == "T5Tokenizer":
    tokenizer = T5Tokenizer.from_pretrained(
                model_name, use_fast=False
             )
else:
    print("[Err] tokenizer_nameの値が不正です。デフォルト設定のAutoTokenizerで実行します。")
    tokenizer = AutoTokenizer.from_pretrained(
                model_name, use_fast=False
             )


### CUDAの検出
if torch.cuda.is_available():
    model = model.to(processor)
else:
    model = model.to("cpu")


### パディングを許可するかどうかを指定
# <参考 : transformersのTokenizerで固定長化する : https://megalodon.jp/2023-0520-1121-10/https://moneyforward-dev.jp:443/entry/2021/10/05/transformers-tokenizer/>
model.config.pad_token_id = tokenizer.eos_token_id


### "<NL>"タグ付与。これがないと正常に返事をしてくれない模様
conversation_history = conversation_list
conversation_history = [f"{uttr['speaker']}: {uttr['text']}" for uttr in conversation_history]
conversation_history = "<NL>".join(conversation_history)


### 会話履歴を渡すとAIの応答を返す関数
def ai_response(input_dialog):

    ### タグをつけて会話の履歴をトークン化
    conversation_history = input_dialog + "<NL>" + f"{ai_name}:"
    input_ids = tokenizer.encode(conversation_history, add_special_tokens=False, return_tensors="pt", padding=True)

    ### モデルに入力を渡して返答を生成
    with torch.no_grad():
        output_ids = model.generate(
            input_ids.to(model.device),
            do_sample=True,
            max_length=token_max_lengh, temperature=token_temperature,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    ### 返答をデコードして整えたら戻す
    response = tokenizer.decode(output_ids.tolist()[0][input_ids.size(1):])
    response = response.replace("<NL>", "\n")
    response = response.replace("</s>", "")

    return response


### （実装予定）忘却関数 - max_lenghが溢れた際の動作
def forget_memory():
    pass


### 会話履歴を表示する関数
def show_conv_list():
    print("")
    print("- - - " * 4 + "会話履歴" + "- - - " * 4 )
    print(conversation_history)
    print("- - - " * 9)
    print("")


### 起動時に設定などを表示する関数
def show_all_configs():

    for i in range(4):
        print("")

    print("<設定項目>")
    print("")
    print(f"モデル名 : {model_name}")
    print(f"トークナイザ名 : {tokenizer_name}")
    print(f"プロセッサ : {processor}")
    print(f"f16モード : {f16_mode}")
    print(f"max_lengh : {token_max_lengh}")
    print(f"temperature : {token_temperature}")
    print("")
    print("")
    print("--- dialog_v3.2 ---")
    print("")
    print("＜オプション＞ （'[]'も入力)")
    print("[break] : 終了")
    print("[clear] : 会話履歴を起動時の状態に戻す")
    print("[remem] : これまでの会話履歴を表示")
    
    ### モデルの保存先
    # デフォルト C:\Users\{ユーザー名}\.cache\huggingface\hub
    print("")
    print("＜モデルの保存先＞")
    print("C:\\Users\\ユーザー名\\.cache\\huggingface\\hub")
    #print(file_utils.default_cache_path)
    
    ### 事前に入力した会話履歴の表示
    show_conv_list()



if __name__ == "__main__":

    show_all_configs()

    ### 初回返答。入力オプションの追加に伴い、対話ループの順序を変更。
    response = ai_response(conversation_history)
    print(f"{ai_name}: " + response)
    conversation_history = conversation_history + "<NL>"+ f"{ai_name}: {response}"
    
    ### 対話ループ
    while True:

        ### ユーザーからの入力を受け取る
        user_input = input(f"{user_name}: ") 

        ### オプション検出
        if user_input.strip() == "":
            print("[Err] 入力が空白です。もう一度入力してください。")
            continue

        elif user_input == "[break]":
            break

        elif user_input == "[clear]":
            conversation_history = conversation_list
            conversation_history = [f"{uttr['speaker']}: {uttr['text']}" for uttr in conversation_history]
            conversation_history = "<NL>".join(conversation_history)

        elif user_input == "[remem]":
            show_conv_list()
            continue

        else:
            ### 入力を会話履歴に追記
            conversation_history = conversation_history + "<NL>" + f"{user_name}: {user_input}"


        ### AIの返答
        response = ai_response(conversation_history)
        print(f"{ai_name}: " + response)
        #subprocess.run("SeikaSay2.exe -cid nnnn -t \"{msg}\"".format(msg=response))

        ### 出力を会話履歴に追記
        conversation_history = conversation_history + "<NL>"+ f"{ai_name}: {response}"
