import asyncio
import requests
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Necessário para ambientes sem GUI
import matplotlib.pyplot as plt
import logging
import os
from telegram import Bot
from telegram.error import TelegramError

# ======================
# CONFIGURAÇÕES
# ======================
BINANCE_API = "https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=15m&limit=500"
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

# Verifica se as variáveis essenciais estão definidas
if not TELEGRAM_TOKEN or not CHAT_ID:
    raise EnvironmentError("❌ Variáveis TELEGRAM_TOKEN e CHAT_ID são obrigatórias. Configure-as no Railway ou .env local.")

bot = Bot(token=TELEGRAM_TOKEN)

# Configuração de logs (arquivo + console)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("bot.log", encoding='utf-8'),
        logging.StreamHandler()  # Mostra logs no terminal também
    ]
)

# Histórico de preços
df = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

# ======================
# FUNÇÃO: pegar dados OHLCV da Binance
# ======================
def get_btc_ohlcv():
    logging.info("[INFO] Consultando OHLCV BTC/USDT na Binance...")
    try:
        response = requests.get(BINANCE_API, timeout=15)
        response.raise_for_status()
        data = response.json()
        processed = []
        for d in data:
            processed.append({
                "timestamp": pd.to_datetime(d[0], unit='ms'),
                "open": float(d[1]),
                "high": float(d[2]),
                "low": float(d[3]),
                "close": float(d[4]),
                "volume": float(d[5])
            })
        df_temp = pd.DataFrame(processed)
        logging.info("[OK] Dados obtidos com sucesso.")
        return df_temp
    except Exception as e:
        logging.error(f"[ERRO] Falha ao pegar OHLCV: {e}")
        return None

# ======================
# FUNÇÃO: enviar mensagem async
# ======================
async def send_message(text):
    try:
        await bot.send_message(chat_id=CHAT_ID, text=text)
        logging.info("[OK] Mensagem enviada no Telegram.")
    except TelegramError as e:
        logging.error(f"[ERRO Telegram] {e}")
        await asyncio.sleep(5)

# ======================
# FUNÇÃO: enviar gráfico se existir
# ======================
async def send_chart_if_exists():
    try:
        with open("btc_chart.png", "rb") as photo:
            await bot.send_photo(chat_id=CHAT_ID, photo=photo)
        logging.info("[OK] Gráfico enviado.")
    except FileNotFoundError:
        logging.error("[ERRO] Gráfico btc_chart.png não encontrado.")
    except TelegramError as e:
        logging.error(f"[ERRO ao enviar gráfico] {e}")

# ======================
# FUNÇÃO: calcular ADX (Average Directional Index)
# ======================
def calculate_adx(df, period=14):
    # True Range
    df['TR'] = df.apply(lambda row: max(
        row['high'] - row['low'],
        abs(row['high'] - row['close']),
        abs(row['low'] - row['close'])
    ), axis=1)

    # Directional Movement
    df['+DM'] = df['high'].diff()
    df['-DM'] = -df['low'].diff()

    df['+DM'] = df['+DM'].where((df['+DM'] > df['-DM']) & (df['+DM'] > 0), 0)
    df['-DM'] = df['-DM'].where((df['-DM'] > df['+DM']) & (df['-DM'] > 0), 0)

    # Smoothed values
    df['TR_smooth'] = df['TR'].rolling(period).mean()
    df['+DI'] = 100 * df['+DM'].rolling(period).mean() / df['TR_smooth']
    df['-DI'] = 100 * df['-DM'].rolling(period).mean() / df['TR_smooth']

    # DX and ADX
    df['DX'] = 100 * abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI'])
    df['ADX'] = df['DX'].rolling(period).mean()

    return df

# ======================
# FUNÇÃO: análise técnica com filtros e confirmações
# ======================
def calculate_indicators(df):
    indicators_triggered = []

    # Verifica tamanho mínimo para cálculos
    if len(df) < 50:
        return [], "neutral", 0.0

    # --- SMA 5, 20, 50, 200 ---
    df["SMA_5"] = df["close"].rolling(5).mean()
    df["SMA_20"] = df["close"].rolling(20).mean()
    df["SMA_50"] = df["close"].rolling(50).mean()
    df["SMA_200"] = df["close"].rolling(200).mean()

    if len(df) >= 20:
        last = df.iloc[-1]
        prev = df.iloc[-2]

        # Cruzamento SMA 5/20
        if prev["SMA_5"] < prev["SMA_20"] and last["SMA_5"] > last["SMA_20"]:
            indicators_triggered.append("SMA Buy")
        elif prev["SMA_5"] > prev["SMA_20"] and last["SMA_5"] < last["SMA_20"]:
            indicators_triggered.append("SMA Sell")

    # --- RSI 14 ---
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df["RSI_14"] = 100 - (100 / (1 + rs))

    if len(df) >= 15:
        last_rsi = df["RSI_14"].iloc[-1]
        if last_rsi < 30:
            indicators_triggered.append("RSI Oversold Buy")
        elif last_rsi > 70:
            indicators_triggered.append("RSI Overbought Sell")

    # --- MACD ---
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    if len(df) >= 27:
        last_macd = df.iloc[-1]
        prev_macd = df.iloc[-2]
        if prev_macd["MACD"] < prev_macd["MACD_signal"] and last_macd["MACD"] > last_macd["MACD_signal"]:
            indicators_triggered.append("MACD Buy")
        elif prev_macd["MACD"] > prev_macd["MACD_signal"] and last_macd["MACD"] < last_macd["MACD_signal"]:
            indicators_triggered.append("MACD Sell")

    # --- Bollinger Bands ---
    df["BB_mid"] = df["close"].rolling(20).mean()
    df["BB_std"] = df["close"].rolling(20).std()
    df["BB_upper"] = df["BB_mid"] + 2 * df["BB_std"]
    df["BB_lower"] = df["BB_mid"] - 2 * df["BB_std"]

    if len(df) >= 20:
        last_close = df["close"].iloc[-1]
        if last_close > df["BB_upper"].iloc[-1]:
            indicators_triggered.append("Bollinger Sell")
        elif last_close < df["BB_lower"].iloc[-1]:
            indicators_triggered.append("Bollinger Buy")

    # --- Volume ---
    df["Volume_avg"] = df["volume"].rolling(20).mean()
    if len(df) >= 20:
        last_vol = df["volume"].iloc[-1]
        avg_vol = df["Volume_avg"].iloc[-1]
        if last_vol > 1.5 * avg_vol:
            indicators_triggered.append("High Volume Alert")

    # --- ADX ---
    df = calculate_adx(df, 14)
    adx_value = df["ADX"].iloc[-1] if len(df) >= 30 else 0.0

    # --- Tendência ---
    trend = "neutral"
    if len(df) >= 200:
        if df["SMA_50"].iloc[-1] > df["SMA_200"].iloc[-1]:
            trend = "bull"
        elif df["SMA_50"].iloc[-1] < df["SMA_200"].iloc[-1]:
            trend = "bear"

    # --- FILTRO: só aceita Buy/Sell se houver tendência compatível OU ADX forte ---
    filtered_indicators = []
    for signal in indicators_triggered:
        if "Buy" in signal:
            if trend == "bull" or adx_value > 25:
                filtered_indicators.append(signal)
        elif "Sell" in signal:
            if trend == "bear" or adx_value > 25:
                filtered_indicators.append(signal)
        else:  # Alertas (volume, etc) passam sempre
            filtered_indicators.append(signal)

    return filtered_indicators, trend, adx_value

# ======================
# FUNÇÃO: gera gráfico (opcional)
# ======================
def generate_chart(df):
    try:
        plt.figure(figsize=(12, 6))
        plt.plot(df["timestamp"], df["close"], label="Close", color='blue', linewidth=2)
        if "SMA_5" in df.columns:
            plt.plot(df["timestamp"], df["SMA_5"], label="SMA 5", color='orange', alpha=0.8)
        if "SMA_20" in df.columns:
            plt.plot(df["timestamp"], df["SMA_20"], label="SMA 20", color='green', alpha=0.8)
        if "SMA_50" in df.columns:
            plt.plot(df["timestamp"], df["SMA_50"], label="SMA 50", color='red', linestyle='--', alpha=0.7)
        if "BB_upper" in df.columns and "BB_lower" in df.columns:
            plt.fill_between(df["timestamp"], df["BB_upper"], df["BB_lower"], color="gray", alpha=0.1, label="Bollinger Bands")
        plt.title("BTC/USDT Análise Técnica (15m)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("btc_chart.png", dpi=150)
        plt.close()
        logging.info("[OK] Gráfico gerado com sucesso.")
    except Exception as e:
        logging.error(f"[ERRO ao gerar gráfico] {e}")

# ======================
# LOOP PRINCIPAL
# ======================
async def main():
    logging.info("🚀 Bot iniciado. Monitorando BTC/USDT (15m)...")
    print("🚀 Bot iniciado. Monitorando BTC/USDT (15m)...")

    while True:
        try:
            df_data = get_btc_ohlcv()
            if df_data is not None:
                global df
                df = df_data

                indicators, trend, adx = calculate_indicators(df)

                # --- FORÇAR SINAL PARA TESTE (descomente para testar envio) ---
                # if not indicators and len(df) > 50:
                #     logging.info("[TESTE] Forçando sinal de compra para validação...")
                #     indicators = ["SMA Buy (teste)", "RSI Oversold Buy (teste)"]
                #     trend = "bull"
                #     adx = 30.0
                # ---

                # Só envia alerta se tiver 2+ indicadores relevantes (exceto volume)
                trade_signals = [s for s in indicators if "Buy" in s or "Sell" in s]
                if len(trade_signals) >= 2:
                    current_price = df['close'].iloc[-1]
                    msg = (
                        f"⚡ ALERTA CONFIRMADO! ({len(trade_signals)} sinais de trade)\n\n"
                        f"Indicadores: {', '.join(indicators)}\n"
                        f"Preço BTC/USDT: ${current_price:,.2f}\n"
                        f"Tendência: {trend.upper()} (SMA50 vs SMA200)\n"
                        f"Força (ADX): {adx:.1f}\n"
                        f"Volume: {'ALTO (alerta)' if 'High Volume Alert' in indicators else 'Normal'}"
                    )
                    await send_message(msg)
                    generate_chart(df)
                    await send_chart_if_exists()
                    logging.info(f"ALERTA ENVIADO: {msg}")
                else:
                    logging.info(f"[INFO] {len(indicators)} indicadores, mas apenas {len(trade_signals)} de trade. Ignorando (mínimo: 2).")

            else:
                logging.warning("[AVISO] Dados não disponíveis. Tentando novamente...")

        except Exception as e:
            logging.critical(f"[ERRO CRÍTICO NO LOOP] {e}")

        await asyncio.sleep(60)  # Checa a cada 1 minuto

# ======================
# INICIALIZAÇÃO
# ======================
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("🛑 Bot interrompido manualmente.")
        print("\n🛑 Bot interrompido manualmente.")
    except Exception as e:
        logging.critical(f"[ERRO FATAL] {e}")
        print(f"[ERRO FATAL] {e}")