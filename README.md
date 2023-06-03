# ***rinna_dialog***


## [ 概要 ]

rinna株式会社が公開しているGPT言語モデル「rinna3.6B」を用いることで、
ローカル環境のCLI上でAIとチャットを出来るようにしたプログラムです。

<br>

動作確認済モデル
* [rinna/japanese-gpt-neox-3.6b](https://huggingface.co/rinna/japanese-gpt-neox-3.6b)
* [rinna/japanese-gpt-neox-3.6b-instruction-sft](https://huggingface.co/rinna/japanese-gpt-neox-3.6b-instruction-sft)
* [rinna/japanese-gpt-neox-3.6b-instruction-sft-v2](https://huggingface.co/rinna/japanese-gpt-neox-3.6b-instruction-sft-v2)
* [rinna/japanese-gpt-neox-3.6b-instruction-ppo](https://huggingface.co/rinna/japanese-gpt-neox-3.6b-instruction-ppo)

![a2](https://github.com/AlgosErgo/rinna_dialog/assets/122419883/7d34f584-2184-489e-9dcf-6594c72a50b0)

<br>

## [ テスト環境 ]

     ### 最小動作環境
     ・CPU : 指定なし
     ・GPU : VRAM 8GB以下では、3発話程度で強制終了。
     ・RAM : 16GB以下では読み込み時にスワップが発生。
     ・ディスク : 15GB以上の空き容量。
                （"model:8GB"+"venv:6GB"）
     
     ### "cuda"指定時のVRAM使用率推移
        - RTX4090, VRAM24GB, max_length=256設定
            アイドル時 : 2.4 GB
            実行時 : 17.6 GB
            5発話でmax_lengthがフロー : 18.1 GB
            レスポンスタイム : 1 ~ 5秒 (体感)

        - TESLA P40, VRAM24GB, max_length=256設定
            アイドル時 : 0 GB
            実行時 : 15.7 GB
            5発話でmax_lengthがフロー : 16.3 GB
            レスポンスタイム : 5 ~ 8秒 (体感)
            備考 : ebayで中古3万円、お得!

        - RTX3080Ti, VRAM12GB, float16指定, max_length=256設定
            アイドル時 : 0 GB
            実行時 : 8.4 GB
            5発話でmax_lengthがフロー :  8.8 GB
            レスポンスタイム : 2 ~ 5秒 (体感)
            備考 : SATA SSDで実行。実行時の読み込みに60秒くらい掛かる模様
            
     ### "cpu"指定時のRAM使用率推移
        - 5950x, RAM64GB, max_length=256設定
            アイドル時 : 8.0 GB
            実行時 : 28.3 GB(ピーク) ～ 21.6 GB(安定)
            5発話でmax_lengthがフロー : 21.7 GB
            レスポンスタイム : 10 ~ 13秒 (体感)
<br>


## [ Requirements ]          
```
- git
- cuda 11.7
- python == 3.10.6
- torch == 1.13.1+cu117
- transformers == 4.29.2
```
<br>

## [ Get Started! ]

### 前提パッケージ

- git_for_windows
  - (https://gitforwindows.org/index.html)
    
- Python3.10.6
  - (https://www.python.org/downloads/release/python-3106/) 
  - (https://www.python.org/ftp/python/3.10.6/python-3.10.6-amd64.exe)

<br>

### 初回起動

1. D:\ > ```git clone https://github.com/AlgosErgo/rinna_dialog```

2. D:\ > ```cd rinna_dialog```

3. D:\rinna_dialog> ```py -3.10 -m venv rinna_venv```

4. D:\rinna_dialog> ```.\rinna_venv\Scripts\activate.bat```

5. (rinna_venv) D:\rinna_dialog> ```pip install -r requirements.txt```

6. (rinna_venv) D:\rinna_dialog> ```python.exe -m pip install --upgrade pip```

7. (rinna_venv) D:\rinna_dialog> ```pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117```

8. (rinna_venv) D:\rinna_dialog> ```python dialog.py```


   ### Tips
- VRAM消費量の削減のためにデフォルトで`Float16`での読み込みとなっています。その影響で環境によっては**起動に60秒以上**かかることが予想されます。
   - dialog.py内の設定項目`F16_mode`より変更可。
- 起動するとHuggingFaceから自動でモデルがダウンロードされ、以下のディレクトリへキャッシュされます。
   - `C:\\Users\\{ユーザー名}\\.cache\\huggingface\\hub`
- ~~モデルを手動でダウンロードした場合は、後述の[Q&A]の手順でキャッシュ化が必要です。~~
   - ローカルに保存したモデルを読み込む際に絶対パスで指定する場合はバックスラッシュを2つ重ねて`\\`とする。
<br>

### 次回以降の起動手順

1. D:\ > ```cd rinna_dialog```

2. D:\rinna_dialog> ```.\rinna_venv\Scripts\activate.bat```

3. (rinna_venv) D:\rinna_dialog> ```python dialog.py```


   ### Tips
- venvを使用せずに、PATHを通したpythonに直接要求パッケージをいれると```python dialog.py```だけで動きます。
    - ただし環境も混ざるので一長一短。
- バッチファイルとかで自動化すると便利。

<br>

## [ 音声合成ソフトとの連携方法 ]

### 1. AssistantSeika - SeikaSay2.exeを使用した方法
![akari_2222](https://github.com/AlgosErgo/rinna_dialog/assets/122419883/1eb6a4c2-aa62-4856-a43d-1b5becf18a69)

#### 必要な物

- Windows OS環境
- voiceroid, voiceroid2, A.I.VOICE, 棒読みちゃん, softalkなどの音声合成ソフト
- AssistantSeika
  - (https://wiki.hgotoh.jp/documents/tools/assistantseika/assistantseika-001a)
- AssistantSeikaに同梱の.\SeikaSay2\SeikaSay2.exe

#### ~~実装手順~~

~~1. コード側に以下の2行を追記~~

V3.4より実装した`[ss2]`オプションにより、dialog.pyを書き換えずに設定が可能になりました。
![3 4](https://github.com/AlgosErgo/rinna_dialog/assets/122419883/d2dd399e-56eb-41ff-a0ed-f14c12f74835)

<br>

#### 使用方法

1, AssistantSeikaSetup.msiを実行してインストール。

2, Voiceroidなどの音声合成ソフトを起動して、最小化はしない。

3, AssistantSeikaを実行し、実行している音声合成ソフトの名前にチェックを入れて「製品スキャン」を実行。

4, スキャンが完了すると自動的にタブが切り替わり、待機状態へ。

<br>

### 2. WindowsでVOICEVOXだけ使う場合

k896951様より情報提供頂いた[voxsay](https://github.com/k896951/voxsay)を使用できます。
同様にsubprocessから渡します。

<br>

## [Q&A]

    Q : なんかエラーが出た！
    A : [Get Started!]を参考に再構築してみましょう。

    Q : 実行すると、`t = torch.tensor([], （略）`と出て固まる！
    A : cudaを使用している場合でもモデルは一度RAMへ読み込まれてからVRAMに展開されます。
        タスクマネージャーを開いてRAMが使われていくのを眺めながら気長に待ちましょう。
        VRAMへの展開が終わっているのに動かない場合はEnterキーを押下すると強制的に先へ進めます。

    Q : ほかのモデルも使える？
    A : モデル名を変更すればHuggingFaceからダウンロードされてキャッシュとして読み込まれますが、正常に動作するかはわかりません。

    Q : モデルのダウンロードが途中で止まってしまう！
    A : モデル名に絶対パスを指定するとローカルモデルを読み込ませることができます。
    
    Q : Linux環境でも動く？
    A : Ubuntu22.04での正常動作は確認できていますが、ほかの環境での動作はわかりません。



## [ 謝辞 ]

AssistantSeika製作者様

k896951様

なんJLLM部の方々
/liveuranus/1678930450/l50


## [ Lisence ]
This project is licensed under the MIT License, see the LICENSE.txt file for details


