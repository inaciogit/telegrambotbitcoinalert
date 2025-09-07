# 📈 Crypto Alert Bot

Bot em Python que monitora BTC/USDT na Binance e envia alertas técnicos (SMA, RSI, MACD, Bollinger) via Telegram quando 2+ indicadores concordam.

## 🚀 Como usar

1. Clone este repositório
2. Instale as dependências: `pip install -r requirements.txt`
3. Configure `TELEGRAM_TOKEN` e `CHAT_ID`
4. Execute: `python crypto_alert_bot.py`

## ⚙️ Indicadores usados

- SMA 5/20 cruzamento
- RSI 14 (sobrevenda/sobrecompra)
- MACD cruzamento
- Bandas de Bollinger
- Filtro de tendência (SMA 50/200)
- ADX para força da tendência

Desenvolvido com ❤️ para traders disciplinados.