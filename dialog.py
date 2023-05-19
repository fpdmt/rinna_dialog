#!/usr/bin/env python3

'''
dialog_v3.1

[概要]
    rinna/japanese-gpt-neox-3.6b-instruction-sft
    https://huggingface.co/rinna/japanese-gpt-neox-3.6b-instruction-sft

    対話型にファインチューニングされたrinna3.6B-instruction-sftを用いることで、
    CLI上でチャットを出来るようにしたプログラムです。


[修正履歴]
  ・ 出力される文書から"</s>"を削除 
  ・ [break]と[reset]を実装
  ・ T5Tokenizer -> AutoTokenizerに変更。(どちらでも動く模様、お好みで変更)
  ・ モデルの保存先を表示
  ・ 設定項目を追加
  ・ float16モードを実装、vram10GBでの動作確認。

  
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
            備考 : ebayで中古3万円、お得！

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
    ・パッケージのverを纏めたrequirements.txtを用意

    > cd C:\\{dialog.pyを保存したフォルダ}
    
    > py -3.10 -m venv LLM_venv (ここでは仮に環境名LLM_venvを作成)
    
    > .\\LLM_venv\\Scripts\\activate.bat

    > pip install -r requirements.txt

    > pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
 
    > python dialog.py

    起動するとHuggingFaceから自動でモデルがダウンロードされキャッシュされる
    モデルを自前でダウンロードした場合は、QAの手順でキャッシュ化が必要


[次回以降起動手順]
   > cd C:\\{dialog_v3を保存したフォルダ}
   > .\\{初回起動時に作成したvenvフォルダ}\\Scripts\\activate.bat
   > python dialog.py

   Tips
     ・venvを使用せずに、pathを通したpythonに直接要求パッケージをいれると、
       python dialog.pyだけで動く。ただし環境も混ざるので一長一短。
     ・バッチファイルとかで自動化すると便利


[小ネタ]
    ボイロなどを持っている人は「AssistantSeika」のSeikaSay2をsubprocesで使うとAIの返答を喋らせることができちゃうゾ
    まだ試してないけどvoicevoxなどのAPIでもワンチャン

[Q&A]
    Q : なんかエラーが出た！
    A : 環境構築メモで再構築してみましょう。それでも動かないときはLLM部まで

    Q : f16モード有効にすると、「t = torch.tensor([], （略）」と出て固まるぞ！
    A : 圧縮に時間が掛かっているようです。
        タスクマネージャーを開いてRAMが使われていくのを眺めて気長に待ちましょう。
        読み込みが終わってるのに動かない場合はEnterキーを押すと強制的に進めます。

    Q : なんかエラーでモデルのダウンロードが途中で止まってしまうんやが・・・
    A : ブラウザでダウンロードした「pytorch_model.bin」をキャッシュの形式にして読み込ませることができます。
'''
#       ・「モデル抜き.zip」をダウンロードして解凍のち以下へ移動
#         -> C:\Users\{users}\.cache\huggingface\hub
#
#       ・ 中身を開いて以下のtxtを探す
#          models--rinna--japanese-gpt-neox-3.6b-instruction-sft\blobs\0c6124c628f8ecc29be1b6ee0625670062340f5b99cfe543ccf049fa90e6207b.txt
#
#       ・ pytorch_model.binを「0c6124c628f8ecc29be1b6ee0625670062340f5b99cfe543ccf049fa90e6207b」にリネーム
#
#       ・ 最後にtxtファイルをゴミ箱へいれて完了
#
#       <完成例>
#        https://i.imgur.com/4Crp8qG.png




################
### 設定項目 ###
################


# モデル名
#
model_name = "rinna/japanese-gpt-neox-3.6b-instruction-sft"

# モデルの移動先
# "cuda" or "cpu"
#
processor = "cuda"


# モデルを圧縮するオプション
#
f16_mode = True


# max_lengthを増やすと会話が長続きする
# ただしvramと要相談
#
token_max_lengh = 1024


# temperatureが低いほど一貫性がある出力を、
# temperatureが高くなるほど多様な出力をするようになる、らしい。
#　<参考 : GPTのtemperatureとは何なのか>
#　https://megalodon.jp/2023-0519-0821-24/https://qiita.com:443/suzuki_sh/items/8e449d231bb2f09a510c
#
token_temperature = 0.7


# 好きな名前を入れてね
#
user_name = "ユーザー"
ai_name = "AI"


# AIの短期記憶リスト(会話履歴)　お好みで書き換え可。
# 対話ループの都合上、AI応答 -> user入力の繰り返しなので、
# listの最後はuserとなるのが好ましい。（もちろんコード側を書き換えてもOK）
#
conversation_list = [
    {"speaker": user_name, "text": "このプログラムはなんですか？"},
    {"speaker": ai_name, "text": "りんなを即席対話型にしたものです"},
    {"speaker": user_name, "text": "プログラムのバグを直すコツを教えてください"},
    {"speaker": ai_name, "text": "動いてさえいればバグなんて直さなくて大丈夫です"},
    {"speaker": user_name, "text": "ほんまに、それでええんか？"},
]




### ここからコード
###################################################################################

import torch
from transformers import T5Tokenizer, AutoTokenizer, AutoModelForCausalLM, file_utils


### 事前学習済みモデルの読み込み
if (f16_mode == True):
    model = AutoModelForCausalLM.from_pretrained(
                 model_name, torch_dtype=torch.float16
             )

else:
    model = AutoModelForCausalLM.from_pretrained(
                model_name
             )


### トークナイザの読み込み
tokenizer = AutoTokenizer.from_pretrained(
                model_name, use_fast=False
             )


### CUDAの検出
if torch.cuda.is_available():
    model = model.to(processor)
else:
    model = model.to("cpu")


### パディングを許可するかどうかを指定
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


### 設定などを起動時に表示する関数
def check_options():

    for i in range(4):
        print("")

    print("<設定項目>")
    print(f"モデル名 : {model_name}")
    print(f"プロセッサ : {processor}")
    print(f"f16モード : {f16_mode}")
    print(f"max_lengh : {token_max_lengh}")
    print(f"temperature : {token_temperature}")

    print("")
    print("")

    print("--- dialog_v3.1 ---")
    print("")
    print("＜オプション＞ （'[]'も入力)")
    print("[break] : 終了")
    print("[clear] : 会話履歴を起動時の状態に戻す")
    
    ### モデルの保存先
    # デフォルト C:\Users\{ユーザー名}\.cache\huggingface\hub
    print("")
    print("＜モデルの保存先＞")
    print("C:\\Users\\ユーザー名\\.cache\\huggingface\\hub")
    #print(file_utils.default_cache_path)
    
    ### 事前に入力した会話履歴の表示
    print("")
    print("- - - " * 4 + "会話履歴" + "- - - " * 4 )
    print(conversation_history)
    print("- - - " * 9)
    print("")



if __name__ == "__main__":

    check_options()
    
    ### 対話ループ
    while True:

        ### AIの返答
        response = ai_response(conversation_history)
        print(f"{ai_name}: " + response)

        ### 出力を会話履歴に追記
        conversation_history = conversation_history + "<NL>"+ f"{ai_name}: {response}"

        ### ユーザーからの入力を受け取る
        user_input = input(f"{user_name}: ") 
        
        ### オプション検出
        if user_input == "[break]":
            break

        elif user_input == "[clear]":
            conversation_history = conversation_list
            conversation_history = [f"{uttr['speaker']}: {uttr['text']}" for uttr in conversation_history]
            conversation_history = "<NL>".join(conversation_history)

        else:
            ### 入力を会話履歴に追記
            conversation_history = conversation_history + "<NL>" + f"{user_name}: {user_input}"

        