from flask import Flask, render_template, request
import pandas as pd
import yfinance as yf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import datetime, timedelta
from io import BytesIO
import base64
import matplotlib
matplotlib.use('Agg')  # Matplotlib'in arka planda çalışması için
import matplotlib.pyplot as plt

app = Flask(__name__)  # Flask uygulamasını başlat

# --- Veri Hazırlığı ---
# Kullanılacak emtiaların ve döviz çiftinin tanımları
commodities = {
    'Altın': 'GC=F',       # Altın
    'Gümüş': 'SI=F',     # Gümüş
    'Platinyum': 'PL=F'    # Platin
}
currency_pair = 'USDTRY=X'  # USD/TRY döviz kuru
start_date = '2016-01-01'  # Verilerin başlangıç tarihi
end_date = datetime.today().strftime('%Y-%m-%d')  # Bugünün tarihi

# Emtia ve döviz verilerini indirme
data_frames = {}
for name, ticker in commodities.items():
    data_frames[name] = yf.download(ticker, start=start_date, end=end_date)
usd_try_data = yf.download(currency_pair, start=start_date, end=end_date)

# Verileri işleme ve gram fiyatı hesaplama
processed_data = {}
for name, df in data_frames.items():
    # Çoklu index sütunları düzleştir
    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    usd_try_data.columns = [col[0] if isinstance(col, tuple) else col for col in usd_try_data.columns]

    # Tarih sütununu eklemek için index'i sıfırla
    df = df.reset_index()
    usd_try = usd_try_data.reset_index()

    # Sadece gerekli sütunları ve eksik değerleri kaldır
    df = df[['Date', 'Close']].dropna()
    df.rename(columns={'Close': f'{name}_Close'}, inplace=True)

    usd_try = usd_try[['Date', 'Close']].dropna()
    usd_try.rename(columns={'Close': 'Close_USDTRY'}, inplace=True)

    # Tarih sütunları üzerinden veri birleştirme
    merged = pd.merge(df, usd_try, on='Date', how='inner')

    # Gerekli sütunların kontrolü ve gram fiyatı hesaplama
    if f'{name}_Close' in merged.columns and 'Close_USDTRY' in merged.columns:
        merged['Gram_Price'] = (merged[f'{name}_Close'] / 31.1) * merged['Close_USDTRY']
        processed_data[name] = merged[['Date', 'Gram_Price']]
    else:
        print(f"Uyarı: {name} için gerekli sütunlar bulunamadı.")


# Grafik oluşturma ve tahmin fonksiyonu
def generate_plot(selected_commodity, forecast_period):
    df = processed_data[selected_commodity]

    # Tahmin periyoduna göre parametreleri belirle
    if forecast_period == "Haftalık":
        past_days = 90  # 3 aylık geçmiş veri
        future_steps = 7  # 7 günlük tahmin
    elif forecast_period == "Aylık":
        past_days = 365  # 1 yıllık geçmiş veri
        future_steps = 30  # 30 günlük tahmin
    elif forecast_period == "Yıllık":
        past_days = 1095  # 3 yıllık geçmiş veri
        future_steps = 365  # 1 yıllık tahmin
    else:
        raise ValueError("Geçersiz tahmin periyodu.")

    # SARIMA modeli ile tahmin
    model = SARIMAX(df['Gram_Price'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
    model_fit = model.fit(disp=False)

    # Geleceğe yönelik tahmin değerleri
    future_index = pd.date_range(start=df['Date'].iloc[-1] + pd.Timedelta(days=1), periods=future_steps)
    forecast_future = model_fit.forecast(steps=future_steps)

    # Grafik başlangıç ve bitiş tarihleri
    start_plot_date = datetime.today() - timedelta(days=past_days)
    end_plot_date = future_index[-1]

    # Grafik oluşturma
    plt.figure(figsize=(14, 7))
    plt.plot(df['Date'], df['Gram_Price'], label='Gerçek Değer', color='gold')
    plt.plot(future_index, forecast_future, label='Gelecek Tahmin', linestyle='--', color='green')
    plt.axvline(x=start_plot_date, color='purple', linestyle='--', label='Grafik Başlangıcı')
    plt.xlim(left=start_plot_date, right=end_plot_date)
    plt.title(f'{selected_commodity} - Gerçek ve Gelecek Tahmin Değerler')
    plt.xlabel('Tarih')
    plt.ylabel('Gram Fiyat (TRY)')
    plt.legend()
    plt.grid()

    # Grafiği base64 formatına çevirme
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plot_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
    buffer.close()
    plt.close()

    return plot_data

# Ana sayfa
@app.route('/')
def index():
    return render_template('index.html')

# Model sayfası
@app.route('/model', methods=['GET', 'POST'])
def model():
    selected_commodity = "Altın"  # Varsayılan emtia
    forecast_period = "Haftalık"  # Varsayılan tahmin periyodu

    if request.method == 'POST':
        # Kullanıcı seçimlerini al
        selected_commodity = request.form.get('commodity')
        forecast_period = request.form.get('forecast_period')

    # Grafik oluştur ve döndür
    plot_url = generate_plot(selected_commodity, forecast_period)
    return render_template('model.html', 
                           selected_commodity=selected_commodity, 
                           forecast_period=forecast_period, 
                           commodities=commodities, 
                           plot_url=plot_url)

# Hakkımızda sayfası
@app.route('/whous')
def whous():
    return render_template('whous.html')

# Giriş sayfası
@app.route('/login')
def login():
    return render_template('login.html')

if __name__ == '_main_':
    app.run(debug=True)