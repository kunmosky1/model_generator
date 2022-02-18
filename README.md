# BFSX2用機械学習データ作成ツール

### ■ このツールについて
このツールはこちらのnoteで公開した手順のうち、「３ー４．学習を行ってモデルを作る」で使う学習データを作成するためのツールです。
https://note.com/kunmosky1/n/n2616f0ecc031

このツールで作成した学習モデル (.zipファイル) があれば、BFSX2に同梱されている mlbot.py を使うことで機械学習の結果に沿った実取引を行うことが出来ます。

### ■ 構成ファイル
#### ●　```model_gererator.ipynb```
学習モデルを作る手順を Jupyter のノートブック形式で用意しました。結果を確認しながら必要な処理をステップバイステップで実行することが出来ます。

#### ●　```model_gererator.py```
全ての一連の手順を実行する場合には、こちらのpythonスクリプトを使うことが出来ます。プロットされた画像や途中経過は指定したwebhookでDiscordへ送ることもできますし、最終的に出来上がったzipファイルに入っている画像やログを確認することもできます。


#### ●　```libs```フォルダ
各種作業を行う実際のコードが入っています。こちらの中身は**BFSX2メンバー限定**で公開します（こちらのリポジトリにはダミーファイルしか入っていません）

#### ●　```candles```フォルダ
ロジック（指値）と特徴量を生成するコードを入れるフォルダです。サンプルとしてrichmanbtc
noteの手順のうち「３ー１．学習のもとになるデータを集める」で用意したデータが入っています。

#### ●　```features```フォルダ
ロジック（指値）と特徴量を生成するコードを入れるフォルダです。サンプルとしてrichmanbtc氏が公開されていたコードが入っています。
noteの手順のうち「３ー２．指値位置を決める」「３ー３．特徴量を考えて作る」に該当する箇所です。

<br>
<br>
<br>

### ■ その他  

最初は様々な plot をするのに時間がかかっていたので、一連の流れを実行させて結果が出たらDisocrdへ画像通知という形で使っていたのですが、プロット部分を高速化したことで、ほとんどのステップが数秒で終わるようになり、個別に確認しながら必要な個所をトライアンドエラーできるJupyter のノートブック形式のほうが使いやすいかなと思いJupyter のノートブック形式でも公開することにしました。  
Jupyterのノートブック形式とpythonスクリプト形式をご自身の使いやすい方、または使用用途などによって使い分けてください。
