#!/usr/bin/env python3

'''
dialog_v3.2


[概要]
    rinna/japanese-gpt-neox-3.6b-instruction-sft
    https://huggingface.co/rinna/japanese-gpt-neox-3.6b-instruction-sft

    対話型にファインチューニングされたrinna3.6B-instruction-sftを用いることで、
    CLI上でAIとチャットを出来るようにしたプログラムです。

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
