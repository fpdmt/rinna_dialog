# ***rinna_dialog***

## [概要]

対話型にファインチューニングされたrinna3.6B-instruction-sftを用いることで、
ローカル環境のCLI上でAIとチャットを出来るようにしたプログラムです。
    
* [rinna/japanese-gpt-neox-3.6b-instruction-sft](https://huggingface.co/rinna/japanese-gpt-neox-3.6b-instruction-sft)
  
![a2](https://github.com/AlgosErgo/rinna_dialog/assets/122419883/7d34f584-2184-489e-9dcf-6594c72a50b0)




## [テスト環境]
     ### cuda vram使用率推移
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
            
     ### cpu RAM使用率推移
        - 5950x, RAM64GB, max_length=256設定
            アイドル時 : 8.0 GB
            実行時 : 28.3 GB(ピーク) ～ 21.6 GB(安定)
            5発話でmax_lengthがフロー : 21.7 GB
            レスポンスタイム : 10 ~ 13秒 (体感)
            
            
## [Requirements]          
```
- cuda 11.7
- python == 3.10.6
- torch == 1.13.1+cu117
- transformers == 4.29.2
```
    
    
## [Get Started!]

   *[仮想環境: Python環境構築ガイド](https://www.python.jp/install/windows/venv.html)
   
### 初回起動
```
D:\>**git clone https://github.com/AlgosErgo/rinna_dialog**
        
D:\>**cd rinna_dialog**

D:\rinna_dialog>**py -3.10 -m venv rinna_venv**

D:\rinna_dialog>**.\rinna_venv\Scripts\activate.bat**

(rinna_venv) D:\rinna_dialog>**pip install -r requirements.txt**

(rinna_venv) D:\rinna_dialog>**python.exe -m pip install --upgrade pip**

(rinna_venv) D:\rinna_dialog>**pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117**
 
(rinna_venv) D:\rinna_dialog>**python dialog.py**
```

   ### Tips
- VRAM削減のためデフォルトでFloat16での読み込みとなっています、その影響で起動には約30秒以上かかることが予想されます。
- 起動するとHuggingFaceから自動でモデルがダウンロードされ以下のディレクトリへキャッシュされます。
   - `C:\\Users\\{ユーザー名}\\.cache\\huggingface\\hub`
- モデルを自前でダウンロードした場合は、後述の[Q&A]の手順でキャッシュ化が必要です。
   
   
### 次回以降の起動手順
```
D:\>**cd rinna_dialog**
         
D:\rinna_dialog>**.\rinna_venv\Scripts\activate.bat****
         
(rinna_venv) D:\rinna_dialog>**python dialog.py**
```

   ### Tips
- venvを使用せずにpath通したpythonに直接要求パッケージをいれるとdialog.pyだけで動きます。ただし環境も混ざるので一長一短。
- バッチファイルとかで自動化すると便利。


## [音声合成ソフトとの連携方法]

### AssistantSeika - SeikaSay2.exeを使用した方法

#### 必要な物

- Windows OS環境
- voiceroid, voiceroid2, A.I.VOICE, 棒読みちゃん, softalkなどの音声合成ソフト
- [AssistantSeika](https://wiki.hgotoh.jp/documents/tools/assistantseika/assistantseika-001a)
- 同梱の.\SeikaSay2\SeikaSay2.exe

#### 実装手順

1. コード側に以下を二行を追記

```python
import subprocess

#対話ループ内の「### AIの返答」を探して、responseに代入し終わったあたりで、subprocessでSeikaSay2.exeに投げる。
#response = ai_response(conversation_history)

#この例だと、A.I.VOICE 紲星あかり
# cidで話者を指定
#
subprocess.run("SeikaSay2.exe -cid 5209 -t \"{msg}\"".format(msg=response))
```

#### 使用方法

1, AssistantSeikaSetup.msiを実行してインストール。

2, Voiceroidなどの音声合成ソフトを起動して、最小化はしない。

3, AssistantSeikaを実行し、実行している音声合成ソフトの名前にチェックを入れて「製品スキャン」を実行。

4, スキャンが完了すると自動的にタブが切り替わる。
   
5,「HTTP機能設定」のタブを開き「HTTP機能を利用する」にチェック。
   
6, 特に変更はせずに右下の「起動する」をクリック、これでlocalhost内で待ち受け状態へ移行。
   サーバを立てたい場合は待ち受けアドレスとポートを変更して、ファイアウォール設定で受信と送信にAssistantseikaのポートを指定して許可。


### WindowsでVOICEVOXだけ使う場合

k896951様より情報提供頂いた[voxsay](https://github.com/k896951/voxsay)を使用できます。
同様にsubprocessから渡します。



## [Q&A]

    Q : なんかエラーが出た！
    A : [Get Started!]を参考に再構築してみましょう。

    Q : 実行すると、`t = torch.tensor([], （略）`と出て固まる！
    A : cudaを使用している場合でもモデルは一度RAMへ読み込まれてからVRAMに展開されます。
        タスクマネージャーを開いてRAMが使われていくのを眺めて気長に待ちましょう。
        読み込みが終わっているのに動かない場合はEnterキーを押すと強制的に先へ進めます。

    Q : モデルのダウンロードが途中で止まってしまう！
    A : ブラウザでダウンロードした`pytorch_model.bin`をキャッシュの形式にして読み込ませることができます。

        ・”models--rinna--japanese-gpt-neox-3.6b-instruction-sft.zip”をダウンロードして解凍のち以下のフォルダに移動。
            -> C:\Users\{users}\.cache\huggingface\hub

        ・中身を開いて以下のtxtを探す。
          models--rinna--japanese-gpt-neox-3.6b-instruction-sft\blobs\0c6124c628f8ecc29be1b6ee0625670062340f5b99cfe543ccf049fa90e6207b.txt

        ・`pytorch_model.bin`を「0c6124c628f8ecc29be1b6ee0625670062340f5b99cfe543ccf049fa90e6207b」にリネーム。

        ・最後にtxtファイルをゴミ箱へいれて完了。

![239520047-196673e6-d4ca-480c-8ae6-299620fc71dc](https://github.com/AlgosErgo/rinna_dialog/assets/122419883/2dfa69e9-5cc8-4172-86b0-543a1d2de697)


## [謝辞]

AssistantSeika製作者様

k896951様

なんJLLM部の方々
/liveuranus/1678930450/l50



