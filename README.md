# ***rinna_dialog***

## [概要]

    対話型にファインチューニングされたrinna3.6B-instruction-sftを用いることで、
    CLI上でAIとチャットを出来るようにしたプログラムです。
    
   * [rinna/japanese-gpt-neox-3.6b-instruction-sft](https://huggingface.co/rinna/japanese-gpt-neox-3.6b-instruction-sft)
  
![dialog_v3 2](https://github.com/AlgosErgo/rinna_dialog/assets/122419883/27207c3c-0bef-4d7d-adb4-450fc41ca890)



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
    
    
## [Get Start!]

   *[仮想環境: Python環境構築ガイド](https://www.python.jp/install/windows/venv.html)
   
   # 初回起動

        `py -3.10 -m venv LLM_venv` (ここでは仮に環境名LLM_venvを作成)
    
         `.\LLM_venv\Scripts\Activate.bat`

         `pip install -r requirements.txt`

         `pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117'
 
         'python dialog.py'

    起動するとHuggingFaceから自動でモデルがダウンロードされキャッシュされます。
   
   
   # 次回以降の起動手順
         `cd C:\{dialog.pyを保存したフォルダ}`
         
         `.\{初回起動時に作成したvenvフォルダ}LLM_venv\Scripts\Activate.bat`
         
         `py thon dialog_v3.py`

   ### Tips
     ・venvを使用せずにpath通したpythonに直接要求パッケージをいれるとdialog_v3.pyだけで動きます。ただし環境も混ざるので一長一短。
     ・バッチファイルとかで自動化すると便利。


## [小ネタ]
    ボイロなどを持っている人は「AssistantSeika」のSeikaSay2をsubprocesで使うとAIの返答を喋らせることができちゃうゾ
    まだ試してないけどvoicevoxなどのAPIでもワンチャン


## [Q&A]

    Q : なんかエラーが出た！
    A : 環境構築メモで再構築してみましょう。それでも動かないときはLLM部まで

    Q : f16モード有効にすると、`t = torch.tensor([], （略）`と出て固まる！
    A : 圧縮に時間が掛かっているようです。
        タスクマネージャーを開いてRAMが使われていくのを眺めて気長に待ちましょう。
        読み込みが終わってるのに動かない場合はEnterキーを押すと強制的に先へ進めます。

    Q : モデルのダウンロードが途中で止まってしまう！
    A : ブラウザでダウンロードした`pytorch_model.bin`をキャッシュの形式にして読み込ませることができます。

        ・”models--rinna--japanese-gpt-neox-3.6b-instruction-sft.zip”をダウンロードして解凍のち以下のフォルダに移動
            -> C:\Users\{users}\.cache\huggingface\hub

        ・中身を開いて以下のtxtを探す
          models--rinna--japanese-gpt-neox-3.6b-instruction-sft\blobs\0c6124c628f8ecc29be1b6ee0625670062340f5b99cfe543ccf049fa90e6207b.txt

        ・`pytorch_model.bin`を「0c6124c628f8ecc29be1b6ee0625670062340f5b99cfe543ccf049fa90e6207b」にリネーム

        ・最後にtxtファイルをゴミ箱へいれて完了

![239520047-196673e6-d4ca-480c-8ae6-299620fc71dc](https://github.com/AlgosErgo/rinna_dialog/assets/122419883/2dfa69e9-5cc8-4172-86b0-543a1d2de697)


