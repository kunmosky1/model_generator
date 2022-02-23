#------------------------------------------------------------------------
#  各種設定
#------------------------------------------------------------------------


# 対象の取引所
#exchange, currency, fee = 'Bybit',  'BTCUSD',    -0.025
#exchange, currency, fee = 'Phemex',  'BTCUSD',    -0.025
#exchange, currency, fee = 'Bitflyer','FX_BTC_JPY', 0.0
exchange, currency, fee = 'Gmo',  'BTC_JPY',    None  # Gmoの場合には個別に期間ごとの手数料を採用します

# 対象の特徴量 (作成済みモデルzipを指定すると作成済みモデルでのバックテストを行う事が出来ます)
calclate_features = 'features/richmanbtc.py'

# 読み込んだローソク足のうちこの時間以降のデータを使用する
startdate = '2000-01-01 00:00:00'

# 通知先Discordの指定
webhook = ''

# ログフォルダ
log_folder = 'logs/'

# テンポラリフォルダ
temp_path = 'temp/'



#------------------------------------------------------------------------
# 必要なライブラリのインポート
#------------------------------------------------------------------------

import importlib.machinery as imm
import os
import pandas as pd
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter('ignore')

# 独自ライブラリのインポート
tools = imm.SourceFileLoader('tools', 'libs/tools.py').load_module()
machine_learning = imm.SourceFileLoader('machine_learning', 'libs/machine_learning.py').load_module()

# 初期化処理
if not 'logger' in locals() :
    logger = tools.Logger(log_folder)
discord = tools.NotifyDiscord(logger, webhook)
model = tools.Model(logger, discord)

# tempフォルダを空にする
try:
    shutil.rmtree(temp_path)
except:
    pass
try:
    if not os.path.exists(temp_path):
        os.makedirs(temp_path)
except:
    pass

# ローソク足データの読込
df_org = machine_learning.load_candles(logger, exchange, currency, fee, startdate)

# ロジックファイル(特徴量ファイル)を読み込む
discord.send( "-"*80 + f"\n{calclate_features}" )
logic = tools.load_logic_file(calclate_features,logger)

# 指値位置の計算
df, target, ref_maker_cost = machine_learning.calc_limit_price(logger, discord, logic, df_org, model, imgfile=temp_path+"order_cost.png")

# 特徴量の計算
df,features = machine_learning.calc_features(logger, logic, df)

if calclate_features.endswith(".py") :

    # 重要度を考慮した特徴量の削除
    delete_features = machine_learning.eliminate_features( logger, discord, df, list(set(features)-set(logic['nouse_columns'])),
                                                           repeat= 30,       # 削除処理を何回まで繰り返すか
                                                           threshold = 2.0,  # 削除する閾値 (小さいほど積極的に削除する 1.5～4.0)
                                                           goal = 0.3,       # 重要度がいくらまで減ったら終了するか
                                                           imgfile = temp_path+"importance.png" )

    # 全区間のクロスバリデーションによる評価
    profit_per_day = machine_learning.cross_validation(logger, discord, model, df,
                          # 採用する特徴量は、ロジックファイルで削除するよう指定されているものと、重要度によって削除することにしたものを除く
                          features = list(set(features) - set(logic['nouse_columns']) - set(delete_features)),
                          y = target,
                          #------------------------------
                          # バリデーションモデルを指定することが出来ます
#                          cvmodel = KFold(),
#                          cvmodel = TimeSeriesSplit(),
                          cvmodel = tools.CombPurgedKFoldCV(n_splits=5, n_test_splits=2, time_gap=int(len(df)/35), embargo_td=pd.Timedelta('500min')),
                          #------------------------------
                          image_path = temp_path )
    print( "全区間でのクロスバリデーションの結果 :  {:.3f}%/日".format(profit_per_day))
    print( "参考：ルールベースでの指値コスト     :  {:.3f}%/日".format(ref_maker_cost))
    print( "CV評価の結果はルールベースでの指値コストよりも増えていないと予測が効果的でないと考えられるので特徴量や削除条件を見直す" )

    # 最終的なモデルの学習期間と評価期間の検討
    split_date = machine_learning.decide_period(logger, discord, df, model, features, target)
    logic['training period'] = model._training_period = str(df.index[0])+' ~ ' + str(split_date)

    # 学習区間で学習モデルの生成
    msplit_date = logic['training period'][28:]
    model.training( df[:split_date], features, target )

    # 作成したモデルを用いて評価区間を予測し、予測と結果の相関関係をプロット
    machine_learning.evaluation( logger, discord, df, model, logic, target, image_path=temp_path )

# バックテスト
split_date = logic['training period'][28:]
test_df, predict = model.evaluation( df[split_date:], report=False ) # 学習モデルで検査期間の評価

if calclate_features.endswith(".py") :
    train_df, predict = model.evaluation( df[:split_date], report=False ) # 学習モデルで学習期間の評価
    machine_learning.backtest_all( logger, discord, train_df, test_df, model, logic, predict, image_path=temp_path, max_pos=99 ) # 全区間でのバックテスト (ポジション上限無し)
    machine_learning.backtest_all( logger, discord, train_df, test_df, model, logic, predict, image_path=temp_path, max_pos=1 )  # 全区間でのバックテスト  (ポジションを上限１に制限した場合）
    machine_learning.backtest_ml_vs_all( logger, discord, test_df, model, logic, predict, image_path=temp_path, max_pos=1 )      # 機械学習の結果に沿ってエントリーした場合と、無条件で全区間エントリーした場合の比較
    machine_learning.backtest_detail( logger, discord, test_df, model, logic, predict, image_path=temp_path, max_pos=logic['params']['pyramiding'] ) # 評価期間の詳細なバックテスト  (ポジションを上限はパラメータで指定した値）

else:
    machine_learning.backtest_ml_vs_all( logger, discord, test_df, model, logic, predict, image_path=temp_path, days=30 )  # 機械学習の結果に沿ってエントリーした場合と、無条件で全区間エントリーした場合の比較
    machine_learning.backtest_detail( logger, discord, test_df, model, logic, predict, image_path=temp_path, days=30, max_pos=logic['params']['pyramiding'] )  # 直近30日の詳細なバックテスト  (ポジションを上限はパラメータで指定した値）

# 完成したモデルを zip ファイルで出力
if calclate_features.endswith(".py") :
    machine_learning.save_model_to_zip(logger, model=model, logic=logic, delete_features=delete_features,
                                       candle_file=exchange+'_'+currency, calclate_features=calclate_features,
                                       temp_path=temp_path)

