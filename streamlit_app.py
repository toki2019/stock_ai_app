import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
# import pandas_ta as ta  # pandas_ta はコメントアウト
import ta # 新しいライブラリをインポート
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
import joblib
import sqlite3

# --- グローバル変数 ---
# AIモデルとスケーラーをStreamlitのセッションステートで管理
# これにより、アプリの状態を維持し、再ロード時にモデルを再利用できます。
if 'AI_MODELS' not in st.session_state:
    st.session_state.AI_MODELS = {} # {ticker: model_object}
if 'SCALERS' not in st.session_state:
    st.session_state.SCALERS = {} # {ticker: scaler_object}
if 'DF_FEATURES_GLOBAL' not in st.session_state:
    st.session_state.DF_FEATURES_GLOBAL = {} # {ticker: df_features}

# モデル保存ディレクトリ
MODEL_DIR = 'ai_models'
# 仮想取引データベースファイル名
DB_VIRTUAL_TRADE = 'virtual_portfolio.db'

# --- 共通設定 ---
# AI学習・バックテストの対象銘柄リスト（複数指定可能）
TARGET_AI_TICKERS = [
    '9984.T', # ソフトバンクグループ
    '7203.T', # トヨタ自動車
    '6758.T', # ソニーG
]

# AIモデル学習用のデータ期間
END_DATE_AI_TRAINING = datetime.now().strftime('%Y-%m-%d')
START_DATE_AI_TRAINING = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d') # 過去5年間のデータ

# 購入可能銘柄表示機能で使用する銘柄リスト (より多くてもOK)
CANDIDATE_STOCKS = [
    {'ticker': '7203.T', 'name': 'トヨタ自動車', 'unit_shares': 100},
    {'ticker': '9984.T', 'name': 'ソフトバンクG', 'unit_shares': 100},
    {'ticker': '6758.T', 'name': 'ソニーG', 'unit_shares': 100},
    {'ticker': '8306.T', 'name': '三菱UFJ', 'unit_shares': 100},
    {'ticker': '4063.T', 'name': '信越化学', 'unit_shares': 100},
    {'ticker': '6098.T', 'name': 'リクルートHD', 'unit_shares': 100},
    {'ticker': 'AAPL', 'name': 'Apple Inc.', 'unit_shares': 1}, # 米国株は通常1株単位
    {'ticker': 'MSFT', 'name': 'Microsoft Corp.', 'unit_shares': 1},
    {'ticker': 'GOOGL', 'name': 'Alphabet Inc. (Class A)', 'unit_shares': 1},
    {'ticker': 'TSLA', 'name': 'Tesla Inc.', 'unit_shares': 1},
    {'ticker': 'F', 'name': 'Ford Motor', 'unit_shares': 1},
]

# --- ユーティリティ関数 ---
@st.cache_data(show_spinner=False, ttl=3600) # データの再取得を避けるためにキャッシュ (1時間)
def get_stock_data(ticker, start_date, end_date):
    """
    Yahoo Financeから株価データを取得します。
    """
    try:
        df = yf.download(ticker, start=start_date, end=end_date, progress=False, show_errors=False)
        
        if df.empty:
            pass
        return df
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()

@st.cache_data(show_spinner=False) # 特徴量計算の再実行を避けるためにキャッシュ
# --- 2. データ前処理と特徴量エンジニアリング ---
def add_features_and_target(df):
    if df.empty:
        return df

    df_copy = df.copy() 
    
    # テクニカル指標の追加 (ta ライブラリの呼び出しを最適化)
    
    # SMA (Simple Moving Average)
    df_copy['SMA_10'] = df_copy['Close'].rolling(window=10).mean() # pandasのrolling().mean()を使用
    df_copy['SMA_20'] = df_copy['Close'].rolling(window=20).mean() # pandasのrolling().mean()を使用

    # RSI (Relative Strength Index)
    # taライブラリを使用するが、より直接的な関数呼び出しを試す
    delta = df_copy['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df_copy['RSI_14'] = 100 - (100 / (1 + rs))

    # MACD (Moving Average Convergence Divergence)
    # taライブラリを使用せず、EMAの組み合わせで直接計算
    exp1 = df_copy['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df_copy['Close'].ewm(span=26, adjust=False).mean()
    df_copy['MACD'] = exp1 - exp2
    df_copy['MACDs'] = df_copy['MACD'].ewm(span=9, adjust=False).mean()
    df_copy['MACDh'] = df_copy['MACD'] - df_copy['MACDs']


    # Bollinger Bands
    # taライブラリを使用せず、直接計算
    window = 20
    std_dev = df_copy['Close'].rolling(window=window).std()
    ma = df_copy['Close'].rolling(window=window).mean()
    df_copy['BBL_20_2.0'] = ma - (std_dev * 2) # Lower Band
    df_copy['BBM_20_2.0'] = ma # Middle Band (SMA)
    df_copy['BBU_20_2.0'] = ma + (std_dev * 2) # Upper Band
    # BBB_20_2.0 と BBP_20_2.0 は省略


    # 価格の変化率 (これは変更なし)
    df_copy['Daily_Return'] = df_copy['Close'].pct_change()
    df_copy['Volume_Change'] = df_copy['Volume'].pct_change()

    # 目的変数 (これは変更なし)
    df_copy['Target'] = (df_copy['Close'].shift(-1) > df_copy['Close']).astype(int) 
    
    df_copy = df_copy.dropna()
    
    return df_copy

# --- モデルの保存・読み込み ---
def save_model(model, scaler, ticker):
    """AIモデルとスケーラーをファイルに保存する"""
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, f'{ticker}_lgbm_model.pkl')
    scaler_path = os.path.join(MODEL_DIR, f'{ticker}_scaler.pkl')
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

def load_model(ticker):
    """ファイルからAIモデルとスケーラーを読み込む"""
    model_path = os.path.join(MODEL_DIR, f'{ticker}_lgbm_model.pkl')
    scaler_path = os.path.join(MODEL_DIR, f'{ticker}_scaler.pkl')
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    return None, None

# --- 3. AI（機械学習）モデルの構築と学習 ---
def train_stock_prediction_model(df_features, ticker):
    """
    LightGBMモデルを学習し、評価します。
    """
    if df_features.empty or 'Target' not in df_features.columns:
        st.error(f"Error: {ticker} のデータが空か 'Target' 列がありません。")
        return None, None

    features = [col for col in df_features.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'Target']]
    X = df_features[features]
    y = df_features['Target']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=features, index=X.index)

    if len(X_scaled_df) < 2: 
        st.error(f"Error: {ticker} のデータが少なすぎます。AIモデルの学習にはより多くのデータが必要です。")
        return None, None

    X_train_val = X_scaled_df.iloc[:-1]
    y_train_val = y.iloc[:-1]

    train_size = int(len(X_train_val) * 0.8)
    X_train, X_test = X_train_val[:train_size], X_train_val[train_size:]
    y_train, y_test = y_train_val[:train_size], y_train_val[train_size:]

    st.info(f"学習データ数: {len(X_train)}, テストデータ数: {len(X_test)}")
    
    model = lgb.LGBMClassifier(objective='binary', random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    st.subheader(f"{ticker} - AIモデル評価レポート (テストデータ)")
    st.write(f"**正解率 (Accuracy):** `{accuracy_score(y_test, y_pred):.4f}`")
    st.text("分類レポート:\n" + classification_report(y_test, y_pred))

    save_model(model, scaler, ticker) # 学習後にモデルを保存
    st.success(f"{ticker} のAIモデルの学習と評価が完了しました！")
    return model, scaler

# --- 4. バックテスト ---
def simple_backtest(ticker, initial_capital=1000000, transaction_cost_rate=0.001, prediction_threshold=0.55):
    """
    学習済みAIモデルとバックテスト用データを使って、簡単なバックテストを行います。
    """
    model = st.session_state.AI_MODELS.get(ticker)
    scaler = st.session_state.SCALERS.get(ticker)
    df_features_global = st.session_state.DF_FEATURES_GLOBAL.get(ticker)

    if model is None or df_features_global is None or scaler is None:
        st.error(f"バックテストを実行できません。{ticker} のAIモデルが学習されていないか、データがありません。")
        return

    features = [col for col in df_features_global.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'Target']]
    
    train_val_len = len(df_features_global.iloc[:-1]) 
    train_size = int(train_val_len * 0.8)
    
    df_for_backtest = df_features_global.iloc[train_size:train_val_len].copy()

    if df_for_backtest.empty:
        st.warning(f"{ticker} のバックテストに利用できるデータがありません。")
        return

    X_test_backtest = df_for_backtest[features]
    X_test_scaled_backtest = scaler.transform(X_test_backtest)
    predictions = model.predict_proba(X_test_scaled_backtest)[:, 1]

    st.subheader(f"{ticker} - バックテスト結果 (閾値: {prediction_threshold})")
    
    capital = initial_capital
    portfolio_value_history = [initial_capital] 
    num_trades = 0
    num_wins = 0
    total_profit = 0
    total_loss = 0

    for i in range(len(predictions)): 
        current_price = df_for_backtest['Close'].iloc[i]
        
        actual_next_day_price = df_for_backtest['Close'].iloc[i] * (1 + df_for_backtest['Daily_Return'].iloc[i+1]) if i + 1 < len(df_for_backtest) else current_price
        
        predicted_proba_up = predictions[i]

        if predicted_proba_up >= prediction_threshold: 
            shares_to_buy = int(capital / current_price) 
            
            if shares_to_buy > 0:
                cost = shares_to_buy * current_price * (1 + transaction_cost_rate)
                
                if capital >= cost: 
                    capital -= cost 
                    
                    revenue = shares_to_buy * actual_next_day_price * (1 - transaction_cost_rate)
                    trade_profit = revenue - (shares_to_buy * current_price * (1 + transaction_cost_rate)) 
                    
                    capital += revenue 

                    num_trades += 1
                    if trade_profit > 0:
                        num_wins += 1
                        total_profit += trade_profit
                    else:
                        total_loss += abs(trade_profit)
                
        portfolio_value_history.append(capital)

    st.write(f"初期資金: ¥{initial_capital:,.0f}")
    st.write(f"最終資金: ¥{capital:,.0f}")
    st.write(f"総損益: ¥{capital - initial_capital:,.0f}")
    st.write(f"取引回数: {num_trades} 回")
    if num_trades > 0:
        st.write(f"勝率: {(num_wins / num_trades) * 100:.2f}%")
        if total_loss > 0:
            st.write(f"プロフィットファクター (総利益 / 総損失): {(total_profit / total_loss):.2f}")
        else:
            st.write("プロフィットファクター: 無限大 (損失なし)")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    plot_dates = df_for_backtest.index[:len(portfolio_value_history)-1] 

    if len(plot_dates) == len(portfolio_value_history[1:]):
        ax.plot(plot_dates, portfolio_value_history[1:], label='Portfolio Value', marker='o', markersize=2)
    else:
        ax.plot(range(len(portfolio_value_history)-1), portfolio_value_history[1:], label='Portfolio Value')
        ax.set_xlabel('Days from start of backtest') 

    ax.axhline(initial_capital, color='r', linestyle='--', label='Initial Capital')
    ax.set_title(f'Portfolio Value Over Time (Backtest for {ticker})')
    ax.set_ylabel('Value (JPY)')
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

# --- 5. 投資金額で購入可能な銘柄表示 ---
@st.cache_data(show_spinner=False, ttl=3600) # 1時間キャッシュ
def get_current_price_single(ticker):
    """
    単一銘柄の現在の株価を取得する (Yahoo Finance)
    """
    try:
        data = yf.download(ticker, period="1d", progress=False, show_errors=False) 
        if not data.empty:
            return data['Close'].iloc[-1]
        else:
            return None
    except Exception as e:
        return None

def display_purchasable_stocks(investment_amount, stock_list):
    """
    投資金額で購入可能な銘柄と、購入可能な単元数を表示する。
    """
    st.subheader(f"投資金額 ¥{investment_amount:,.0f} で購入可能な銘柄")
    
    found_purchasable = False
    results = []

    progress_text = "銘柄情報を取得中..."
    my_bar = st.progress(0, text=progress_text)
    
    for i, stock_info in enumerate(stock_list):
        ticker = stock_info['ticker']
        name = stock_info['name']
        unit = stock_info.get('unit_shares', 100) 

        current_price = get_current_price_single(ticker)

        if current_price is None:
            my_bar.progress((i + 1) / len(stock_list), text=f"スキップ中: {name} ({ticker}) - データ取得失敗")
            continue

        cost_per_unit = current_price * unit
        
        if investment_amount >= cost_per_unit:
            purchasable_units = int(investment_amount // cost_per_unit)
            total_cost = purchasable_units * cost_per_unit
            remaining_funds = investment_amount - total_cost

            results.append({
                '銘柄名': name,
                'ティッカー': ticker,
                '現在値': f"¥{current_price:,.2f}",
                '単元株数': unit,
                '単元費用': f"¥{cost_per_unit:,.0f}",
                '購入可能単元': purchasable_units,
                '合計費用': f"¥{total_cost:,.0f}",
                '残資金': f"¥{remaining_funds:,.0f}"
            })
            found_purchasable = True
        my_bar.progress((i + 1) / len(stock_list), text=f"処理中: {name} ({ticker})")

    my_bar.empty() 
    
    if not found_purchasable:
        st.info("現在の投資金額で購入可能な銘柄は見つかりませんでした。より少額で買える銘柄を追加するか、投資金額を増やしてみてください。")
    else:
        df_results = pd.DataFrame(results)
        st.dataframe(df_results.set_index('銘柄名'))

# --- 6. 最新株価予測 (AIモデルが学習済みの場合) ---
def predict_latest_stock_price(ticker):
    """
    学習済みAIモデルを使用して、最新の株価予測を行います。
    """
    model = st.session_state.AI_MODELS.get(ticker)
    scaler = st.session_state.SCALERS.get(ticker)

    if model is None or scaler is None:
        st.error(f"{ticker} のAIモデルが学習されていません。先に「AIモデル学習」を実行してください。")
        return

    recent_data_start = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d') 
    recent_df = get_stock_data(ticker, recent_data_start, datetime.now().strftime('%Y-%m-%d'))

    if recent_df.empty:
        st.error(f"最新の株価データを取得できませんでした: {ticker}")
        return

    recent_df_features = add_features_and_target(recent_df)

    if recent_df_features.empty:
        st.error("最新データから特徴量を生成できませんでした。データの期間が短すぎるか、欠損値が多い可能性があります。")
        return

    latest_features_row = recent_df_features.iloc[[-1]].copy()
    
    if 'Target' in latest_features_row.columns:
        latest_features_row = latest_features_row.drop(columns=['Target']) 

    features = [col for col in latest_features_row.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume']]
    
    if not all(f in latest_features_row.columns for f in features):
        st.error("特徴量の一部が最新データに存在しません。モデルとデータ生成ロジックを確認してください。")
        return

    if latest_features_row[features].isnull().any().any():
        st.error("最新データの特徴量にNaNが含まれています。予測できません。")
        return

    latest_data_scaled = scaler.transform(latest_features_row[features])
    
    prediction_proba = model.predict_proba(latest_data_scaled)[:, 1][0]
    prediction_label = "上がる" if prediction_proba >= 0.5 else "下がるか横ばい"
    
    st.subheader(f"最新株価予測 ({ticker})")
    st.write(f"現在の株価データ最終日: `{latest_features_row.index[-1].strftime('%Y-%m-%d')}`")
    st.write(f"**翌営業日の株価は `{prediction_label}` と予測 (確率: `{prediction_proba:.2f}`)**")
    
    if prediction_proba >= 0.55: 
        st.success("→ 買い推奨シグナル (予測確率が閾値以上のため)")
    elif prediction_proba <= 0.45: 
        st.warning("→ 売り推奨シグナル (予測確率が閾値以下のため)")
    else:
        st.info("→ 様子見シグナル (予測に明確な傾向なし)")

# --- 7. 仮想投資機能 ---
def init_virtual_trade_db():
    """仮想取引データベースを初期化し、取引履歴テーブルを作成する。
    同時に、仮想ポートフォリオ（保有銘柄）テーブルも作成する。
    """
    conn = sqlite3.connect(DB_VIRTUAL_TRADE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS virtual_trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trade_date TEXT NOT NULL,
            ticker TEXT NOT NULL,
            action TEXT NOT NULL, -- 'BUY', 'SELL', 'HOLD', 'ERROR', 'SKIP'
            price REAL,          -- 取引時の価格 (購入/売却)
            shares INTEGER,      -- 取引株数
            cost_or_revenue REAL, -- 購入費用または売却収益 (手数料込み)
            predicted_proba REAL, -- 予測確率 (買いの場合)
            actual_next_day_change REAL, -- 実際の翌日変化率 (損益確定用)
            confirmed_profit_loss REAL -- 確定損益 (売却時のみ)
        )
    ''')
    # 仮想ポートフォリオ（保有銘柄）テーブル
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS virtual_portfolio (
            ticker TEXT PRIMARY KEY,
            shares INTEGER NOT NULL,
            avg_price REAL NOT NULL, -- 平均取得価格
            current_value REAL,      -- 現在の評価額
            last_update_date TEXT    -- 最終更新日
        )
    ''')
    conn.commit()
    conn.close()

def record_virtual_trade(trade_date, ticker, action, price=None, shares=None, cost_or_revenue=None, predicted_proba=None, actual_next_day_change=None, confirmed_profit_loss=None):
    """仮想取引をデータベースに記録する。"""
    conn = sqlite3.connect(DB_VIRTUAL_TRADE)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO virtual_trades (trade_date, ticker, action, price, shares, cost_or_revenue, predicted_proba, actual_next_day_change, confirmed_profit_loss) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (trade_date, ticker, action, price, shares, cost_or_revenue, predicted_proba, actual_next_day_change, confirmed_profit_loss)
    )
    conn.commit()
    conn.close()

def update_virtual_portfolio(ticker, shares_change, price, current_date_str):
    """仮想ポートフォリオの保有銘柄情報を更新する。"""
    conn = sqlite3.connect(DB_VIRTUAL_TRADE)
    cursor = conn.cursor()

    cursor.execute("SELECT shares, avg_price FROM virtual_portfolio WHERE ticker = ?", (ticker,))
    current_holding = cursor.fetchone()

    if current_holding:
        current_shares, current_avg_price = current_holding
        new_shares = current_shares + shares_change

        if new_shares == 0: # 全て売却した場合
            cursor.execute("DELETE FROM virtual_portfolio WHERE ticker = ?", (ticker,))
        elif shares_change > 0: # 追加購入
            new_total_cost = (current_shares * current_avg_price) + (shares_change * price)
            new_avg_price = new_total_cost / new_shares
            cursor.execute("UPDATE virtual_portfolio SET shares = ?, avg_price = ?, last_update_date = ? WHERE ticker = ?",
                           (new_shares, new_avg_price, current_date_str, ticker))
        else: # 一部売却 (平均価格は変わらない)
            cursor.execute("UPDATE virtual_portfolio SET shares = ?, last_update_date = ? WHERE ticker = ?",
                           (new_shares, current_date_str, ticker))
    else: # 新規購入
        cursor.execute("INSERT INTO virtual_portfolio (ticker, shares, avg_price, last_update_date) VALUES (?, ?, ?, ?)",
                       (ticker, shares_change, price, current_date_str))
    
    conn.commit()
    conn.close()