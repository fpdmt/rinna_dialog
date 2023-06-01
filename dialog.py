#!/usr/bin/env python3

'''
dialog_v3.5
https://github.com/AlgosErgo/rinna_dialog

[概要]
    rinna/japanese-gpt-neox-3.6b-instruction-sft
    https://huggingface.co/rinna/japanese-gpt-neox-3.6b-instruction-sft

    対話型にファインチューニングされたrinna3.6B-instruction-sftを用いることで、
    CLI上でAIとチャットを出来るようにしたプログラムです。

[修正予定]
・generation configuration fileエラーメッセージを解決


'''


#################
###  設定項目  ###
#################



#\\\\\\\ [ ユーザー設定 ] \\\\\\

####
# 好きな名前を入れてね
#
user_name = "ユーザー"
ai_name = "AI"


####
# AIの短期記憶リスト(会話履歴)　お好みで書き換え可。
# 対話ループの都合上、user入力 -> AI返答の繰り返しなので、
# listの最後はAI返答となるのが好ましい（もちろんコード側を書き換えてもOK）
# (V3.3より変更)
#
conversation_list = [
    {"speaker": user_name, "text": "あなたは誰ですか？"},
    {"speaker": ai_name, "text": "私はAIアシスタントです。何かお手伝いできることはありますか？"},
]



#\\\\\\\ [ 文章生成設定 ] \\\\\\

####
# temperatureが低いほど一貫性がある出力を、高くなるほど多様な出力をするようになる。
#　<参考 : GPTのtemperatureとは何なのか : https://megalodon.jp/2023-0519-0821-24/https://qiita.com:443/suzuki_sh/items/8e449d231bb2f09a510c>
#
token_temperature = 0.7


####
# max_lengthを増やすと会話が長続きする。
# ※ただしvramと要相談。
#
token_max_lengh = 1024


####
# 会話履歴を保持する数。
#
max_conv_list = 10


###
# AI返答の最後にユーザーのセリフが含まれている場合は削除する。
#
skip_response = True


###
# 長文生成の設定。
# 読点'。'を区切りとして、以下で設定した値以上の文が生成された場合はそれ以降を削除する。
#
sentence_limit = 5


### パディングを許可するかどうかを指定。
#  <参考 : transformersのTokenizerで固定長化する : https://megalodon.jp/2023-0520-1121-10/https://moneyforward-dev.jp:443/entry/2021/10/05/transformers-tokenizer/>
#   - True  : 旧rinna 3.6b (instruction-sft, japanese-gpt-neox-3.6b)
#   - False : 新rinna 3.6b (instruction-ppo, instruction-sft-v2)
#
token_padding = False


### AI出力のオウム返し対策。（すでに生成された単語や文脈に属する単語にペナルティ）
#  <参考 : generate()のパラメータ : https://note.com/npaka/n/n5d296d8ae26d>
#  <参考 : transformer doc : https://huggingface.co/transformers/v2.11.0/main_classes/model.html>
#   - 1.0 : default
#   - 1.1 : 新rinna 3.6b (instruction-ppo, instruction-sft-v2)
#
token_repetition_penalty = 1.1



#\\\\\\\ [ モデルの読み込み ] \\\\\\

####
# モデルの移動先。
# "cuda"か"cpu"を指定。
#
processor = "cuda"


####
# モデルを圧縮するオプション。
#   - True  : 圧縮    【vram使用率:約8GB】
#   - False : 未圧縮  【vram使用率:約16GB】
#
f16_mode = True


####
# トークナイザ名。
# "AutoTokenizer" or "T5Tokenizer"
#
tokenizer_name = "AutoTokenizer"


####
# モデル名。
#   ・ローカルモデルを参照する際は、絶対パスで記述。
#       - (例: "D:\\models\\japanese-gpt-neox-3.6b-instruction-ppo")
#   ・動作確認済モデル
#       - rinna/japanese-gpt-neox-3.6b  (token_repetition_penalty=1.0, token_padding=True)
#       - rinna/japanese-gpt-neox-3.6b-instruction-sft  (token_repetition_penalty=1.0, token_padding=True)
#       - rinna/japanese-gpt-neox-3.6b-instruction-sft-v2
#       - rinna/japanese-gpt-neox-3.6b-instruction-ppo
#
model_name = "rinna/japanese-gpt-neox-3.6b-instruction-sft-v2"


####
# モデルコンフィグファイル。
#
#model_config_name = "config.json"



#\\\\\\\ [ AssistantSeika ] \\\\\\

###
# SeikaSay2.exeの連携。
#
ss2_state = False


###
# SeikaSay2.exeの保存先。
#
ss2_proc = "SeikaSay2.exe"


###
# SeikaSay2 - cidの指定。
#
ss2_cid = "00000"




### ここからコード。
###################################################################################

import os
import subprocess
import torch
from transformers import T5Tokenizer, AutoTokenizer, AutoModelForCausalLM



### 事前学習済みモデルの読み込み。
if f16_mode == True:
    model = AutoModelForCausalLM.from_pretrained(
                 model_name, 
                 torch_dtype=torch.float16
            )
elif f16_mode == False:
    model = AutoModelForCausalLM.from_pretrained(
                 model_name, 
            )
else:
    print("[Err] f16_modeの値が不正です。デフォルト設定のfloat16モードで実行します。")
    model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                torch_dtype=torch.float16
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
    

### CUDAの検出。
if torch.cuda.is_available():
    model = model.to(processor)
else:
    model = model.to("cpu")


### padding設定。
if token_padding:
    model.config.pad_token_id = tokenizer.eos_token_id


### "<NL>"タグ付与。これがないと正常に返事をしてくれない模様。
conversation_history = conversation_list
conversation_history = [f"{uttr['speaker']}: {uttr['text']}" for uttr in conversation_history]
conversation_history = "<NL>".join(conversation_history)


### 会話履歴を渡すとAIの応答を返す関数。
def ai_response(input_dialog):

    ### タグをつけて会話の履歴をトークン化。
    conversation_history = input_dialog + "<NL>" + f"{ai_name}:"
    input_ids = tokenizer.encode(conversation_history, add_special_tokens=False, return_tensors="pt", padding=token_padding)

    ### モデルに入力を渡して返答を生成。
    with torch.no_grad():
        output_ids = model.generate(
            input_ids.to(model.device),
            do_sample=True,
            repetition_penalty=token_repetition_penalty,
            max_length=token_max_lengh, temperature=token_temperature,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    ### 返答をデコード。
    response = tokenizer.decode(output_ids.tolist()[0][input_ids.size(1):])

    ### 返答にユーザーの入力が含まれている場合は削除
    if skip_response:
        if f"{user_name}: " in response:
            response = response.split(f"{user_name}: ")[0]

    ### 長文である場合は、sentence_limit値以上の読点で以降の文章を削除。
    indexes = [i for i, char in enumerate(response) if char == "。"]
    if sentence_limit <= len(indexes):
        index = indexes[sentence_limit - 1]
        response = response[:index + 1]

    response = response.replace("<NL>", "\n")
    response = response.replace("</s>", "")

    return response


### 忘却関数 - max_conv_listが溢れた際に古い会話履歴から1つずつ削除。
#
def forget_conv_list(input_conv_history):
    conversation_list = input_conv_history.split("<NL>")
    if len(conversation_list) > max_conv_list:
        # print("[info] flagged")
        # "<NL>" を探して古い会話を1つ削除する
        index = input_conv_history.find("<NL>")
        if index != -1:
            ret_conv_list = input_conv_history[index+len("<NL>"):]
    else:
        ret_conv_list = input_conv_history

    return ret_conv_list


### SeikaSay2.exe設定。
#
def update_ss2_state():
    print("[ss2] SeikaSay2 連携状態")
    print("")
    print("< 設定 >")
    print("1 : True")
    print("2 : False")
    print("")
    print("< 現在の状態 >")
    print(f"{ss2_state}")
    print("")

    while True:
        input_state = input("入力 : ")

        if not input_state.isdigit():
            print("[Err] : 数字で入力してください。")
            continue
        elif input_state == "1":
            return True
        elif input_state == "2":
            return False
        else:
            print("[Err] : 不正な値です。")
            continue


def update_ss2_proc(current):
    print("[ss2] exeプログラムの参照。入力せずにEnterキーで変更なし。")
    input_proc = input("保存先 : ")
    if input_proc == "":
        return current
    else:
        return input_proc


def update_ss2_cid():
    print("[ss2] CharacterIDの変更")
    while True:
        input_cid = input("CIDを入力 : ")
        if not input_cid.isdigit():
            print("[Err] : CIDは数字で入力してください。")
            continue
        if user_input.strip() == "":
            print("[Err] : 入力が空白です。もう一度入力してください。")
            continue
        else:
            cid_str = str(input_cid)
            return int(cid_str)


### SeikaSay2設定を表示する関数
def show_ss2_config():
    print("")
    print("< SeikaSay2 設定 >")
    print("< 現在の設定 >")
    print(f"連携状態 : {ss2_state}")
    print(f"SeikaSay2.exe保存先 : {ss2_proc}")
    print(f"CharacterID : {ss2_cid}")
    print("")


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

    print("< 設定項目 >")
    print("")
    print(f"モデル名 : {model_name}")
    print(f"トークナイザ名 : {tokenizer_name}")
    print(f"プロセッサ : {processor}")
    print(f"Float16圧縮モード : {f16_mode}")
    print(f"max_lengh : {token_max_lengh} / max_conv_list : {max_conv_list}")
    print(f"temperature : {token_temperature} / repetition_penalty : {token_repetition_penalty}")
    print(f"skip_response : {skip_response} / sentence_limit : {sentence_limit}")
    show_ss2_config()
    print("")
    print("")
    print("--- dialog_v3.5 ---")
    print("")
    print("< オプション > （'[]'も入力)")
    print("[break] : 終了")
    print("[clear] : 会話履歴を起動時の状態に戻す")
    print("[remem] : これまでの会話履歴を表示")
    print("[ss2] : SeikaSay2で音声を再生するキャラクターを変更")
    print("")
    print("＜モデルの保存先＞")
    print("C:\\Users\\ユーザー名\\.cache\\huggingface\\hub")
    #print(file_utils.default_cache_path)
    
    ### 事前に入力した会話履歴の表示
    show_conv_list()




if __name__ == "__main__":

    show_all_configs()
    
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
            continue

        elif user_input == "[remem]":
            show_conv_list()
            continue

        elif user_input == "[ss2]":
            show_ss2_config()
            print("")
            ss2_state = update_ss2_state()
            ss2_proc = update_ss2_proc(ss2_proc)
            ss2_cid = update_ss2_cid()
            for i in range(4):
                print("")
            print("[ss2] 設定が更新されました。")
            show_ss2_config()
            continue

        else:
            ### 入力を会話履歴に追記
            conversation_history = conversation_history + "<NL>" + f"{user_name}: {user_input}"


        ### AIの返答
        response = ai_response(conversation_history)
        print(f"{ai_name}: " + response)
        print("")

        if ss2_state:
            args =  f"\"{ss2_proc}\""
            args += f" -cid \"{ss2_cid}\""
            args += f" -t \"{response}\""
            subprocess.run(args)

        ### 出力を会話履歴に追記
        conversation_history = conversation_history + "<NL>"+ f"{ai_name}: {response}"

        ### 会話が増えすぎたら古い履歴から削除
        conversation_history = forget_conv_list(conversation_history)
        
