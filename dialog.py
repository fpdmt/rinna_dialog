# -*- coding: utf-8 -*-
#!/usr/bin/env python3

'''
dialog_v3.6
https://github.com/AlgosErgo/rinna_dialog

[概要]
    rinna japanese-gpt-neox-3.6b-instruction-sft
    https://huggingface.co/rinna/japanese-gpt-neox-3.6b-instruction-sft

    対話型にファインチューニングされたrinna3.6B-instruction-sftを用いることで、
    CLI上でAIとチャットを出来るようにしたプログラムです。

    
[修正履歴]
・UI周りの最適化。モデル読み込み時のメッセージを追加。
・新オプションを追加
    - [regen] : 会話履歴から一番新しいAI返答を削除して、再生成する。
    - [force] : AI返答文をユーザー入力で上書きする。
    - [ng] : ユーザー入力からNGリストを作成。AI返答にNGワードが含まれている場合は削除する。
・[regen]と[force]実装に伴い、関数:forget_conv_listに再生成モードを追加。
・skip_responseをskip_user_resに変更

'''


#################
###  設定項目  ###
#################



#========= [ ユーザー設定 ] =========

####
# 好きな名前を入れてね
#
user_name = "ユーザー"
ai_name = "AI"


####
# AIの短期記憶リスト(会話履歴)　お好みで書き換え可。
#   追記する場合は次のように追記する。
#       {"speaker": user_name, "text": "ここにユーザー入力"},
#       {"speaker": ai_name, "text": "ここにAI返答を入力"},
#
# (以下V3.3より変更)
# 対話ループの都合上、user入力 -> AI返答の繰り返しなので、
# listの最後はAI返答となるのが好ましい（もちろんコード側を書き換えてもOK）
#
conversation_list = [
    {"speaker": user_name, "text": "あなたは誰ですか？"},
    {"speaker": ai_name, "text": "私はAIアシスタントです。何かお手伝いできることはありますか？"},
]


####
# 会話履歴で保持する会話数。溢れた場合は一番古い履歴から1つずつ削除していく。
#
max_conv_list = 30


####
# 会話履歴の読み込み準備。
#
# [clear]オプションで復元するために会話履歴を再代入
conversation_history = conversation_list
# "<NL>"タグ付与。これがないと正常に返事をしてくれない模様。
conversation_history = [f"{uttr['speaker']}: {uttr['text']}" for uttr in conversation_history]
conversation_history = "<NL>".join(conversation_history)



#========= [ 文章生成設定 ] =========

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
# AI返答の最後にユーザーのセリフが含まれている場合は削除する。
#
skip_user_res = True


####
# 長文生成の設定。
# 読点'。'を区切りとして、以下で設定した値以上の文が生成された場合はそれ以降を削除する。
#
sentence_limit = 5


####
#  パディングを許可するかどうかを指定。
#  <参考 : transformersのTokenizerで固定長化する : https://megalodon.jp/2023-0520-1121-10/https://moneyforward-dev.jp:443/entry/2021/10/05/transformers-tokenizer/>
#   (23.06.03訂正: 新旧ともにTrueで良い模様。）
#
token_padding = True


#### 
# AI出力のオウム返し対策。（すでに生成された単語や文脈に属する単語にペナルティ）
#  <参考 : generate()のパラメータ : https://note.com/npaka/n/n5d296d8ae26d>
#  <参考 : transformer doc : https://huggingface.co/transformers/v2.11.0/main_classes/model.html>
#   - 1.0 : default
#   - 1.1 : 新rinna 3.6b (instruction-ppo, instruction-sft-v2)
#
token_repetition_penalty = 1.1



#========= [ NGワードの設定 ] =========

#### 
# AI返答から完全一致で[]削除するNGワードリスト。
# ai_responceとupdate_ng_listで参照。
#
ng_words = [ "@@dummy1", "@@dummy2"
            
]

#### 
# 削除したNGワードを表示する。
#   - True : 表示する。
#   - False : 表示しない。
#
show_ng_word = False


#========= [ モデルの読み込み ] =========

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


#========= [ AssistantSeika ] =========

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

import subprocess
import torch
from transformers import T5Tokenizer, AutoTokenizer, AutoModelForCausalLM



### 事前学習済みモデルの読み込み。
if f16_mode == True:
    print(
        "\n\n  [info] 言語モデルの読み込みを開始。( 推定所要時間 : 60-120秒 )" + "\n    < Float16=True >\n\n" +
        "  [tips]\n" + "   ・ CUDAを使用している場合でも、モデルは一度RAMに読込まれた後にVRAMへ展開されます。\n\n" +"   ・ 読み込みの進捗状況はタスクマネージャなどで確認できます。\n\n" + "   ・ VRAMへの展開が終わっているのに先に進まない場合は、Enterキーを押下すると強制的に実行できます。\n\n\n\n"
    )
    model = AutoModelForCausalLM.from_pretrained(
                 model_name, 
                 torch_dtype=torch.float16
    )
elif f16_mode == False:
    print("\n\n  [info] モデルの読み込みを開始。" +
          "\n<Float16=False>\n\n"
    )
    model = AutoModelForCausalLM.from_pretrained(
                 model_name
    )
else:
    print("\n\n  [Err] f16_modeの値が不正です。" +
          "デフォルト設定のfloat16モードで実行。\n\n"
    )
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
    print("\n\n[Err] tokenizer_nameの値が不正です。" +
          "デフォルト設定のAutoTokenizerで実行。\n\n"
    )
    tokenizer = AutoTokenizer.from_pretrained(
                    model_name, use_fast=False
    )


### 初期化処理
#
def init():
    ### CUDAの検出とモデルの移動。
    if torch.cuda.is_available():
        model.to(processor)
    else:
        model.to("cpu")
    ### padding設定。
    if token_padding:
        model.config.pad_token_id = tokenizer.eos_token_id


### 会話履歴を渡すとAIの応答を返す関数。
#
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
    if skip_user_res:
        if f"{user_name}: " in response:
            response = response.split(f"{user_name}: ")[0]

    ### 返答にng_wordsに含まれている要素がある場合は削除
    for ng_word in ng_words:
        if ng_word in response:
            response = response.replace(ng_word, "")
            if show_ng_word:
                print(f"\n  [NG] NGワード「{ng_word}」を削除しました。\n")

    ### 長文である場合は、sentence_limit以上の数の読点で以降の文章を削除。
    indexes = [i for i, char in enumerate(response) if char == "。"]
    if sentence_limit <= len(indexes):
        index = indexes[sentence_limit - 1]
        response = response[:index + 1]
    response = response.replace("<NL>", "\n")
    response = response.replace("</s>", "")

    return response


def forget_conv_list(input_conv_history, mode=None):
    ### 再生成モード
    if mode == "regen":
        ### "<NL>" を探して「一番新しい会話」を1つ削除する
        index = input_conv_history.rfind("<NL>")
        if index != -1:
            ret_conv_list = input_conv_history[:index]
        else:
            ### 例外処理: "<NL>" が見つからない場合
            raise ValueError("[Err] <NL>が見つかりません")
    ### 通常モード
    else:
        conv_list = input_conv_history.split("<NL>")
        if len(conv_list) > max_conv_list:
            ### "<NL>" を探して「一番古い会話」を1つ削除する
            index = input_conv_history.find("<NL>")
            if index != -1:
                ret_conv_list = input_conv_history[index + len("<NL>"):]
            else:
                raise ValueError("[Err] <NL>が見つかりません")
        else:
            ret_conv_list = input_conv_history

    return ret_conv_list


### NG処理
#
def update_ng_list():
    print("  NGワードを追加します。[exit]を入力すると中止できます。")
    print("    複数のワードを入力する場合は、カンマと半角スペースで区切ってください。")
    while True:
        new_words = input("  [NG] 追加のNGワードを入力（中止 [exit]）: ")
        if new_words == "[exit]":
            break
        else:
            new_ng_words = new_words.split(", ")
            for new_word in new_ng_words:
                if new_word not in ng_words:
                    ng_words.append(new_word)
                    print(f"NGワードリストに「{new_word}」を追加しました。")
                    show_ng_list()
                else:
                    print(f"NGワードリストに「{new_word}」は既に存在します。")
                    show_ng_list()
                    continue


### SeikaSay2.exe設定。
#
def update_ss2_state():
    print(
        "  [ss2] SeikaSay2 連携状態\n\n"+
        "    < 設定 >\n"+
        "     1 : True\n" + "     2 : False\n" +"\n\n"+
        "    < 現在の状態 >\n" + f"{ss2_state}\n\n"
    )
    while True:
        input_state = input(" 入力[1-2] : ")

        if not input_state.isdigit():
            print("  [Err] : 数字で入力してください。")
            continue
        elif input_state == "1":
            return True
        elif input_state == "2":
            return False
        else:
            print("  [Err] : 不正な値です。")
            continue


def update_ss2_proc(current):
    print("  [ss2] exeプログラムの参照。入力せずにEnterキーで変更なし。")
    input_proc = input("  保存先 : ")
    if input_proc == "":
        return current
    else:
        return input_proc


def update_ss2_cid():
    print("  [ss2] CharacterIDの変更")
    while True:
        input_cid = input("  CIDを入力 : ")
        if not input_cid.isdigit():
            print("  [Err] : CIDは数字で入力してください。")
            continue
        if user_input.strip() == "":
            print("  [Err] : 入力が空白です。もう一度入力してください。")
            continue
        else:
            cid_str = str(input_cid)
            return int(cid_str)


### SeikaSay2設定を表示する関数
def show_ss2_config():
    print("\n < SeikaSay2 設定 >" +
    "\n  < 現在の設定 >" +
    f"\n    連携状態 : {ss2_state}" +
    f"\n    SeikaSay2.exe保存先 : {ss2_proc}" +
    f"\n    CharacterID : {ss2_cid}\n"
    )



### 会話履歴を表示する関数
def show_conv_list():
    print(
        "\n" + "- - - " * 4 + "会話履歴" + "- - - " * 4 +
        "\n" + f"  {conversation_history}"+
        "\n" + "- - - " * 9 + "\n"
    )


### NGワードを表示
def show_ng_list():
    print("\n  < NGワード一覧 >"+
    f"{ng_words}\n"
    )



# 起動時に設定などを表示する関数
def show_all_configs():
    print(
        "\n\n\n\n" +
        " < 設定項目 >\n" +
        f"  モデル名 : {model_name}\n" + f"  トークナイザ名 : {tokenizer_name}\n" +
        f"  プロセッサ : {processor}  /  Float16圧縮モード : {f16_mode}\n" +
        f"  max_lengh : {token_max_lengh}  /  max_conv_list : {max_conv_list}\n" +
        f"  temperature : {token_temperature}  /  repetition_penalty : {token_repetition_penalty}\n" +
        f"  skip_user_res : {skip_user_res}  /  sentence_limit : {sentence_limit}\n"
    )
    ### SeikaSay2設定の表示
    show_ss2_config()
    print(
        "\n\n --- dialog_v3.6 ---\n""\n"+
        " < コマンドオプション > （'[]'も入力)\n" + "   [clear] : 会話履歴を起動時の状態に戻す。\n"+
        "   [remem] : これまでの会話履歴を表示。\n" + "   [force] : AI返答をユーザー入力で上書きする。\n"+
        "   [regen] : 一番新しいAI返答を削除して再生成。\n" + "      [ng] : AI返答をフィルターするNGワードを設定。\n"+
        "     [ss2] : SeikaSay2で音声を再生するキャラクターを変更。\n\n" + " ＜モデルの保存先＞\n" + " C:\\Users\\ユーザー名\\.cache\\huggingface\\hub\n"
    )



if __name__ == "__main__":
    ### 初期化
    init()
    show_all_configs()
    show_conv_list()

    ### 対話ループ
    while True:
        ### ユーザーからの入力を受け取る
        user_input = input(f"{user_name}: ") 

        ### オプション検出
        if user_input.strip() == "":
            print("[Err] 入力が空白です。もう一度入力してください。")
            continue

        elif user_input == "[break]" or user_input == "[exit]":
            break

        elif user_input == "[clear]":
            conversation_history = conversation_list
            conversation_history = [f"{uttr['speaker']}: {uttr['text']}" for uttr in conversation_history]
            conversation_history = "<NL>".join(conversation_history)
            continue

        elif user_input == "[remem]":
            show_conv_list()
            continue

        elif user_input == "[force]":
            conversation_history = forget_conv_list(conversation_history, "regen")
            print("\n\n  [force] : 一番新しいAI返答を上書きします。")
            force_input = input(f"  [force] {ai_name}: ") 
            conversation_history = conversation_history + "<NL>" + f"{ai_name}: {force_input}"
            continue

        elif user_input == "[regen]":
            conversation_history = forget_conv_list(conversation_history, "regen")

        elif user_input == "[ss2]":
            show_ss2_config()
            print("")
            ss2_state = update_ss2_state()
            ss2_proc = update_ss2_proc(ss2_proc)
            ss2_cid = update_ss2_cid()
            for i in range(4):
                print("")
            print("   [ss2] 設定が更新されました。")
            show_ss2_config()
            continue

        elif user_input == "[ng]":
            update_ng_list()
            continue

        elif user_input == "[conf]":
            show_all_configs()
            continue

        else:
            ### 入力を会話履歴に追記
            conversation_history = conversation_history + "<NL>" + f"{user_name}: {user_input}"

        ### AIの返答
        response = ai_response(conversation_history)
        print(f"{ai_name}: " + response)
        print("")

        ### 連携状態がTrueの場合はSeikaSay2で再生。
        if ss2_state:
            args =  f"\"{ss2_proc}\""
            args += f" -cid \"{ss2_cid}\""
            args += f" -t \"{response}\""
            subprocess.run(args)

        ### 返答を会話履歴に追記
        conversation_history = conversation_history + "<NL>"+ f"{ai_name}: {response}"
        ### 会話が増えすぎたら古い履歴から削除
        conversation_history = forget_conv_list(conversation_history)
        
