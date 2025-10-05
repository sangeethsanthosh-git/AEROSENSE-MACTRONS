from flask import Flask, render_template, request, session, flash, redirect, url_for
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import warnings
import requests
from datetime import datetime, timedelta
import folium
from flask_babel import Babel
from flask import jsonify
import re
import difflib
import unicodedata

# -------- Text normalization and tokenization --------
_TOKEN_RE = re.compile(r"""
    (https?://\S+|           # URLs
     [\w.+-]+@[\w-]+\.\w+|   # emails
     [#@][\w_]+|             # hashtags/mentions
     \w+['’]\w+|             # contractions like don't, it’s
     \w+(?:-\w+)+|           # hyphenated words like long-term
     \w+|                    # plain words/numbers
     [\u2600-\u27BF]|        # misc symbols
     [\U0001F300-\U0001FAFF] # emojis (requires wide build)
    )
""", re.UNICODE | re.VERBOSE)

def normalize_text(s: str) -> str:
    try:
        return unicodedata.normalize('NFKC', s or '').strip().lower()
    except Exception:
        return (s or '').strip().lower()

def tokenize(s: str):
    s = normalize_text(s)
    try:
        toks = _TOKEN_RE.findall(s)
    except Exception:
        toks = s.split()
    return [t for t in toks if t.strip()]

def fuzzy_contains(tokens, choices, threshold=0.8):
    # Accepts short and long tokens; uses exact or fuzzy match
    for t in tokens:
        for c in choices:
            if t == c:
                return True
            if len(t) >= 2 and difflib.SequenceMatcher(None, t, c).ratio() >= threshold:
                return True
    return False

def any_token(tokens, choices):
    cs = set(choices)
    return any(t in cs for t in tokens)

def extract_city_phrase(text: str):
    # Supports "in/at/for/near/around <city>", multi-word names
    m = re.search(r'\b(?:in|at|for|near|around)\s+([a-zA-Z][a-zA-Z .-]{1,40})', text, re.IGNORECASE)
    if m:
        return m.group(1).strip(' .-')
    return None

def classify_short_long_words(tokens):
    alpha = [t for t in tokens if any(ch.isalpha() for ch in t)]
    short = [t for t in alpha if len(t) <= 2]
    longw = [t for t in alpha if len(t) >= 8]
    return {
        "short": sorted(set(short)),
        "long": sorted(set(longw))
    }

warnings.filterwarnings('ignore')

app = Flask(__name__)
app.secret_key = '31dc051843b6c8ba7f4a770a2b3237f8'  # Replace securely

# i18n
app.config['BABEL_DEFAULT_LOCALE'] = 'en'
app.config['BABEL_SUPPORTED_LOCALES'] = ['en', 'hi', 'ml', 'ta']

def select_locale():
    lang = session.get('lang')
    if not lang:
        lang = request.args.get('lang')
        if lang:
            session['lang'] = lang
    return lang or app.config.get('BABEL_DEFAULT_LOCALE', 'en')

babel = Babel(app, locale_selector=select_locale)

# Units helpers and template injection
def _to_celsius(t, units):
    try:
        return (t - 32) * 5/9 if units == 'imperial' else t
    except Exception:
        return t

def _mph_to_mps(v, units):
    try:
        return v * 0.44704 if units == 'imperial' else v
    except Exception:
        return v

@app.context_processor
def inject_units():
    u = session.get('units', 'metric')
    return {
        "units": u,
        "units_symbol": "°F" if u == "imperial" else "°C",
        "wind_unit": "mph" if u == "imperial" else "m/s"
    }

# Preference helpers and date/time filters
def pref(key, default=None):
    return session.get(key, default)

def pref_set(key, value):
    session[key] = value

@app.template_filter("fmt_time")
def fmt_time(ts):
    tf = pref("time_format", "24h")
    try:
        dt = datetime.strptime(ts, "%Y-%m-%d %H:%M")
    except Exception:
        try:
            dt = datetime.fromisoformat(ts)
        except Exception:
            return ts
    return dt.strftime("%b %d, %I:%M %p") if tf == "12h" else dt.strftime("%b %d, %H:%M")

@app.template_filter("fmt_date")
def fmt_date(dstr):
    dfmt = pref("date_format", "DD-MM-YYYY")
    try:
        dt = datetime.strptime(dstr, "%Y-%m-%d")
    except Exception:
        return dstr
    return dt.strftime("%d-%m-%Y") if dfmt == "DD-MM-YYYY" else dt.strftime("%m-%d-%Y")

print(f"Current pandas version: {pd.__version__}")

# Preprocessing
def preprocessing(df):
    indian_cities = [
        'Ahmedabad','Aizawl','Amaravati','Amritsar','Bengaluru','Bhopal','Brajrajnagar',
        'Chandigarh','Chennai','Coimbatore','Delhi','Ernakulam','Gurugram','Guwahati',
        'Hyderabad','Jaipur','Jorapokhar','Kochi','Kolkata','Lucknow','Mumbai','Patna',
        'Shillong','Talcher','Thiruvananthapuram','Visakhapatnam','Nagaon','Silchar','Byrnihat'
    ]
    df = df[df['city'].isin(indian_cities)]
    df = df.rename(columns={'city': 'City', 'last_update': 'Date', 'OZONE': 'O3'})
    df_pivot = df.pivot_table(
        index=['City','Date','station','country','state','latitude','longitude'],
        columns='pollutant_id', values='pollutant_avg', aggfunc='first'
    ).reset_index()
    if 'OZONE' in df_pivot.columns:
        df_pivot = df_pivot.rename(columns={'OZONE':'O3'})

    required = ['PM2.5','PM10','NO2','O3','CO','SO2']
    missing = [p for p in required if p not in df_pivot.columns]
    if missing:
        df_pivot['AQI'] = np.nan
        df_pivot['AQI_Bucket'] = np.nan
    else:
        def calc_si(c, lo, hi, ilo, ihi):
            if pd.isna(c): return None
            c = float(c)
            if c <= lo: return ilo
            if c >= hi: return ihi
            return ((c - lo)/(hi - lo))*(ihi - ilo)+ilo
        bps = {
            'PM2.5': [(0,30,0,50),(31,60,51,100),(61,90,101,200),(91,120,201,300),(121,250,301,400),(251,float('inf'),401,500)],
            'PM10':  [(0,50,0,50),(51,100,51,100),(101,250,101,200),(251,350,201,300),(351,430,301,400),(431,float('inf'),401,500)],
            'NO2':   [(0,40,0,50),(41,80,51,100),(81,180,101,200),(181,280,201,300),(281,400,301,400),(401,float('inf'),401,500)],
            'O3':    [(0,50,0,50),(51,100,51,100),(101,168,101,200),(169,208,201,300),(209,748,301,400),(749,float('inf'),401,500)],
            'CO':    [(0,1.0,0,50),(1.1,2.0,51,100),(2.1,10.0,101,200),(10.1,17.0,201,300),(17.1,34.0,301,400),(34.1,float('inf'),401,500)],
            'SO2':   [(0,40,0,50),(41,80,51,100),(81,380,101,200),(381,800,201,300),(801,1600,301,400),(1601,float('inf'),401,500)]
        }
        sub_cols = []
        for p in required:
            if p in df_pivot.columns:
                col = f'{p}_sub_index'
                df_pivot[col] = df_pivot[p].apply(lambda x: None if pd.isna(x) else max(
                    [calc_si(x,lo,hi,ilo,ihi) for lo,hi,ilo,ihi in bps[p] if calc_si(x,lo,hi,ilo,ihi) is not None] or [None]
                ))
                sub_cols.append(col)
        df_pivot['AQI'] = df_pivot[sub_cols].max(axis=1) if sub_cols else np.nan
        def bucket(aqi):
            try:
                aqi = float(aqi)
                if aqi <= 50: return 'Good'
                if aqi <= 100: return 'Satisfactory'
                if aqi <= 200: return 'Moderate'
                if aqi <= 300: return 'Poor'
                if aqi <= 400: return 'Very Poor'
                return 'Severe'
            except Exception:
                return None
        df_pivot['AQI_Bucket'] = df_pivot['AQI'].apply(bucket)
        df_pivot = df_pivot.drop(columns=sub_cols, errors='ignore')

    expected = ['City','Date','station','PM2.5','PM10','NO','NO2','NOx','NH3','CO','SO2','O3','Benzene','Toluene','AQI','AQI_Bucket','Year','Month']
    for c in expected:
        if c not in df_pivot.columns: df_pivot[c] = np.nan
    df_pivot['Year'] = np.nan
    df_pivot['Month'] = np.nan
    return df_pivot[expected]

def load_pickle(file_path, use_pandas=False, critical=True):
    try:
        if use_pandas: return pd.read_pickle(file_path)
        with open(file_path,'rb') as f: return pickle.load(f)
    except AttributeError as e:
        if ('_unpickle_block' in str(e) or 'new_block' in str(e)) and use_pandas:
            pest = pd.read_csv(r'city_air\city1_day.csv')
            data = preprocessing(pest.copy()); data.to_pickle(file_path); return data
        if not critical: return None
        raise

def get_playing_condition(weather, aqi_bucket, sport):
    temp = weather.get('temp', 25)
    wind_speed = weather.get('wind_speed', 0)
    desc = weather.get('description', '').lower()
    if sport == 'cricket':
        if aqi_bucket in ['Severe','Very Poor'] or 'rain' in desc or wind_speed > 20: return "Poor"
        if aqi_bucket in ['Poor'] or temp > 35 or 'storm' in desc: return "Caution"
        return "Good"
    if sport == 'football':
        if aqi_bucket in ['Severe'] or temp > 35 or 'storm' in desc: return "Poor"
        if aqi_bucket in ['Very Poor','Poor'] or 'heavy rain' in desc: return "Caution"
        return "Good"
    return "N/A"

def get_aqi_for_city(city):
    city_data = df[df['City'].str.lower() == city.lower()]
    if not city_data.empty:
        latest = city_data.sort_values('Date', ascending=False).iloc[0]
        return latest['AQI'], latest['AQI_Bucket']
    return None, None

def _uvi_level(u):
    if u is None: return ("N/A","N/A")
    try: u = float(u)
    except Exception: return ("N/A","N/A")
    if u <= 2: return ("Low","green")
    if u <= 5: return ("Moderate","yellow")
    if u <= 7: return ("High","orange")
    if u <= 10: return ("Very High","red")
    return ("Extreme","violet")

def _heat_index_celsius(t_c, rh):
    try:
        t_f = t_c*9/5+32
        hi_f = -42.379 + 2.04901523*t_f + 10.14333127*rh - 0.22475541*t_f*rh \
             - 0.00683783*(t_f**2) - 0.05481717*(rh**2) + 0.00122874*(t_f**2)*rh \
             + 0.00085282*t_f*(rh**2) - 0.00000199*(t_f**2)*(rh**2)
        return (hi_f-32)*5/9
    except Exception:
        return None

def _rain_intensity(mm_per_hr):
    if mm_per_hr is None: return "none"
    try: r = float(mm_per_hr)
    except Exception: return "none"
    if r < 0.5: return "very light"
    if r < 4.0: return "moderate"
    if r < 8.0: return "heavy"
    return "violent"

def _build_advice(weather, aqi_val, uvi_now):
    temp = weather.get("temp", 25)     # °C
    rh   = weather.get("humidity", 50)
    wind = weather.get("wind_speed", 0) # m/s
    desc = (weather.get("description") or "").lower()
    rain_1h = weather.get("rain",{}).get("1h") if isinstance(weather.get("rain"),dict) else None
    hi = _heat_index_celsius(temp, rh)

    outdoor = "OK"
    try: aqi = float(aqi_val) if aqi_val is not None else None
    except Exception: aqi = None
    if aqi and aqi > 100: outdoor = "Modify"
    if hi is not None and hi >= 32: outdoor = "Modify"
    if hi is not None and hi >= 41: outdoor = "Avoid"
    try:
        if uvi_now is not None and float(uvi_now) >= 8: outdoor = "Modify"
    except Exception: pass
    if wind >= 14: outdoor = "Caution" if outdoor == "OK" else outdoor

    agri = "Field work OK"
    if "thunder" in desc or "storm" in desc: agri = "Postpone"
    if rain_1h is not None and _rain_intensity(rain_1h) in ["heavy","violent"]: agri = "Postpone"

    watering = "Maybe"
    if (aqi and aqi > 150) or (isinstance(uvi_now,(int,float)) and uvi_now >= 8 and temp >= 32 and rh <= 40):
        watering = "Likely"
    return {"heat_index_c": round(hi,1) if hi is not None else "N/A","outdoor":outdoor,"agri":agri,"watering":watering}

def get_weather_forecast(city=None, api_key="31dc051843b6c8ba7f4a770a2b3237f8", lat=None, lon=None, display_city=None):
    api_units = session.get('units', 'metric')
    api_lang  = session.get('lang', 'en')

    def process_weather_data(data, city_name):
        if data.get("cod") == 200:
            T = data["main"]["temp"]              # display units
            H = data["main"]["humidity"]
            V = data["wind"]["speed"]             # display units
            T_c   = _to_celsius(T, api_units)
            V_mps = _mph_to_mps(V, api_units)
            current_time_utc = datetime.utcnow().timestamp()
            tz = data.get("timezone",0)
            sunrise = data["sys"]["sunrise"]; sunset = data["sys"]["sunset"]
            is_day = sunrise <= current_time_utc + tz <= sunset
            description = data["weather"][0]["description"]
            out = {
                "city": city_name, "temp": round(T), "description": description, "humidity": H,
                "wind_speed": V, "model_inputs":[T_c, T_c+2, T_c-2, 1013.25, H, 10 if "clear" in description.lower() else 5, V_mps, V_mps+1],
                "is_day": is_day
            }
            if "rain" in data: out["rain"] = data["rain"]
            out["_norm"] = {"temp_c": T_c, "wind_mps": V_mps}
            return out
        return {"city": city_name, "temp": 25 if api_units=='metric' else 77, "description": "unknown",
                "humidity":50, "wind_speed":0, "model_inputs":[25,27,23,1013.25,50,5,0,1],
                "is_day":True, "_norm":{"temp_c":25,"wind_mps":0}}

    def process_short_term_forecast(data, city_name):
        if data.get("cod") == "200":
            out=[]
            for item in data["list"][:5]:
                date = datetime.fromtimestamp(item["dt"]).strftime("%Y-%m-%d %H:%M")
                temp = round(item["main"]["temp"])
                desc = item["weather"][0]["description"]
                out.append({"date":date,"temp":temp,"description":desc})
            return out
        return []

    def process_daily_forecast(data, city_name):
        if data.get("cod") == "200":
            daily={}
            for item in data["list"]:
                date = datetime.fromtimestamp(item["dt"]).strftime("%Y-%m-%d")
                temp = round(item["main"]["temp"]); desc=item["weather"][0]["description"]
                if date not in daily or temp > daily[date]["temp"]:
                    daily[date] = {"temp":temp,"description":desc}
            return [{"date":d,"temp":v["temp"],"description":v["description"]} for d,v in list(daily.items())[:5]]
        return []

    fallback_cities = {"kattakkada":"Thiruvananthapuram"}

    if lat is not None and lon is not None:
        coord_url    = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units={api_units}&lang={api_lang}"
        forecast_url = f"http://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={api_key}&units={api_units}&lang={api_lang}"
        oc_url       = f"https://api.openweathermap.org/data/3.0/onecall?lat={lat}&lon={lon}&appid={api_key}&units={api_units}&exclude=minutely,alerts&lang={api_lang}"
        try:
            current_data = requests.get(coord_url).json()
            weather_data = process_weather_data(current_data, display_city if display_city else current_data.get("name","Unknown Location"))
            forecast_data = requests.get(forecast_url).json()
            weather_data["forecast"] = process_short_term_forecast(forecast_data, weather_data["city"])
            weather_data["daily_forecast"] = process_daily_forecast(forecast_data, weather_data["city"])
            try:
                oc = requests.get(oc_url, timeout=10).json()
                uvi_now = oc.get("current",{}).get("uvi"); uvi_daily=[d.get("uvi") for d in oc.get("daily",[])][:3]
            except Exception:
                uvi_now, uvi_daily = None, []
            name,color = _uvi_level(uvi_now)
            weather_data["uvi"] = {"now": uvi_now, "daily": uvi_daily, "level": name, "color": color}
            norm = dict(weather_data); norm["temp"]=weather_data["_norm"]["temp_c"]; norm["wind_speed"]=weather_data["_norm"]["wind_mps"]
            weather_data["advice"] = _build_advice(norm, None, uvi_now)
            return weather_data
        except Exception as e:
            print("Weather by coordinates error:", e)
            return process_weather_data({"cod":404}, display_city or "Unknown Location")

    if city:
        original_city = city; cl = city.lower()
        if cl in fallback_cities:
            city = fallback_cities[cl]
            print(f"Fallback city '{original_city}' -> '{city}'")
        current_url  = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units={api_units}&lang={api_lang}"
        forecast_url = f"http://api.openweathermap.org/data/2.5/forecast?q={city}&appid={api_key}&units={api_units}&lang={api_lang}"
        try:
            current_data = requests.get(current_url).json()
            weather_data = process_weather_data(current_data, original_city if cl in fallback_cities else city)
            forecast_data = requests.get(forecast_url).json()
            weather_data["forecast"] = process_short_term_forecast(forecast_data, weather_data["city"])
            weather_data["daily_forecast"] = process_daily_forecast(forecast_data, weather_data["city"])
            return weather_data
        except Exception as e:
            print("Weather by city error:", e)
            flash(f"Error fetching weather data for {original_city}. Please try again later.", "error")
            return process_weather_data({"cod":500}, original_city)

    flash("Unable to fetch weather data. Please try again with a different city.", "error")
    return process_weather_data({"cod":500}, "Unknown Location")

# Load air quality data and models
print("Loading city1_day.csv...")
pest = pd.read_csv(r'city_air\city1_day.csv')
df = preprocessing(pest.copy())
cities = df['City'].unique()
try:
    model      = load_pickle('model.pkl', critical=False)
    pest_model = load_pickle('classifier.pkl', critical=False)
    pest_data  = load_pickle('index.pkl', use_pandas=True, critical=True)
    pest_solution = load_pickle('solution.pkl', use_pandas=True, critical=False)
    modelFor   = load_pickle('tree_gridcv.pkl', critical=True)
except Exception as e:
    print("Model load warning:", e)

# Venues
try:
    venues_df = pd.read_csv('venues.csv'); stadiums = venues_df.to_dict('records')
except FileNotFoundError:
    stadiums = []

# Routes
@app.route("/", methods=['GET','POST'])
def home():
    weather_data = None
    if request.method == 'POST':
        if 'city' in request.form and request.form['city'].strip():
            city = request.form['city'].strip()
            weather_data = get_weather_forecast(city=city)
            if weather_data:
                session['city'] = city; session['weather_data'] = weather_data
                session['weather_timestamp'] = datetime.now().isoformat()
                session['geolocation_attempted'] = True
        elif 'lat' in request.form and 'lon' in request.form:
            lat = float(request.form['lat']); lon = float(request.form['lon'])
            geocode_url = f"http://api.openweathermap.org/geo/1.0/reverse?lat={lat}&lon={lon}&limit=1&appid=31dc051843b6c8ba7f4a770a2b3237f8"
            city = None
            try:
                geocode_data = requests.get(geocode_url).json()
                if geocode_data: city = geocode_data[0]['name']
            except Exception: pass
            if pref('location_precision','precise') == 'city':
                city = session.get('default_city') or city
                if city:
                    weather_data = get_weather_forecast(city=city)
            else:
                if city:
                    weather_data = get_weather_forecast(city=city, lat=lat, lon=lon, display_city=city)
            if weather_data:
                session['city'] = city; session['weather_data'] = weather_data
                session['weather_timestamp'] = datetime.now().isoformat()
                session['geolocation_attempted'] = True
        elif 'geolocation_failed' in request.form:
            session['geolocation_attempted'] = True
            flash("Geolocation failed. Please search for a city manually.", "error")
    else:
        if 'weather_timestamp' in session and 'weather_data' in session:
            last = datetime.fromisoformat(session['weather_timestamp'])
            if datetime.now() - last < timedelta(seconds=10):
                weather_data = session['weather_data']
            else:
                session['geolocation_attempted'] = False
    if not session.get('geolocation_attempted', False) and weather_data is None:
        session['geolocation_attempted'] = False
    return render_template("home.html",
        weather=weather_data,
        short_term_forecast=weather_data.get("forecast", []) if weather_data else [],
        daily_forecast=weather_data.get("daily_forecast", []) if weather_data else [],
        geolocation_attempted=session.get('geolocation_attempted', False)
    )

@app.route("/table", methods=['GET','POST'])
def table():
    weather_data = None
    if request.method == 'POST' and 'city' in request.form and request.form['city'].strip():
        city = request.form['city'].strip()
        weather_data = get_weather_forecast(city=city)
        if weather_data:
            session['city']=city; session['weather_data']=weather_data
            session['weather_timestamp']=datetime.now().isoformat()
    elif 'weather_data' in session and 'weather_timestamp' in session:
        last = datetime.fromisoformat(session['weather_timestamp'])
        if datetime.now() - last < timedelta(minutes=3):
            weather_data = session['weather_data']
        else:
            session.pop('weather_data',None); session.pop('weather_timestamp',None); session.pop('city',None)
    with open("Airquality_index.csv") as file:
        reader = csv.reader(file); header = next(reader)
        return render_template("table.html", header=header, rows=reader, weather=weather_data)

@app.route('/air', methods=['GET','POST'])
def datas():
    weather_data = None
    if request.method == 'POST' and 'city' in request.form and request.form['city'].strip():
        city = request.form['city'].strip()
        weather_data = get_weather_forecast(city=city)
        if weather_data:
            session['city']=city; session['weather_data']=weather_data; session['weather_timestamp']=datetime.now().isoformat()
    elif 'weather_data' in session and 'weather_timestamp' in session:
        last = datetime.fromisoformat(session['weather_timestamp'])
        if datetime.now() - last < timedelta(minutes=3):
            weather_data = session['weather_data']
        else:
            session.pop('weather_data',None); session.pop('weather_timestamp',None); session.pop('city',None)
    return render_template('indexx.html', weather=weather_data)

@app.route('/index', methods=['GET','POST'])
def index():
    weather_data = None; AQI_predict = None
    if request.method == 'POST' and 'city' in request.form and request.form['city'].strip():
        city = request.form['city'].strip()
        weather_data = get_weather_forecast(city=city)
        if weather_data:
            session['city']=city; session['weather_data']=weather_data; session['weather_timestamp']=datetime.now().isoformat()
            if 'model_inputs' in weather_data and 'modelFor' in globals():
                AQI_predict = modelFor.predict([weather_data['model_inputs']])
    elif 'weather_data' in session and 'weather_timestamp' in session:
        last = datetime.fromisoformat(session['weather_timestamp'])
        if datetime.now() - last < timedelta(minutes=3):
            weather_data = session['weather_data']
            if 'model_inputs' in weather_data and 'modelFor' in globals():
                AQI_predict = modelFor.predict([weather_data['model_inputs']])
        else:
            session.pop('weather_data',None); session.pop('weather_timestamp',None); session.pop('city',None)
    return render_template('result.html', weather=weather_data, prediction=AQI_predict)

@app.route("/predictions", methods=['GET','POST'])
def predictions():
    weather_data = None
    if request.method == 'POST' and 'city' in request.form and request.form['city'].strip():
        city = request.form['city'].strip()
        weather_data = get_weather_forecast(city=city)
        if weather_data:
            session['city']=city; session['weather_data']=weather_data; session['weather_timestamp']=datetime.now().isoformat()
    elif 'weather_data' in session and 'weather_timestamp' in session:
        last = datetime.fromisoformat(session['weather_timestamp'])
        if datetime.now() - last < timedelta(minutes=3):
            weather_data = session['weather_data']
        else:
            session.pop('weather_data',None); session.pop('weather_timestamp',None); session.pop('city',None)
    return render_template("prediction.html", weather=weather_data)

@app.route("/about", methods=['GET','POST'])
def about():
    weather_data = None
    if request.method == 'POST' and 'city' in request.form and request.form['city'].strip():
        city = request.form['city'].strip()
        weather_data = get_weather_forecast(city=city)
        if weather_data:
            session['city']=city; session['weather_data']=weather_data; session['weather_timestamp']=datetime.now().isoformat()
    elif 'weather_data' in session and 'weather_timestamp' in session:
        last = datetime.fromisoformat(session['weather_timestamp'])
        if datetime.now() - last < timedelta(minutes=3):
            weather_data = session['weather_data']
        else:
            session.pop('weather_data',None); session.pop('weather_timestamp',None); session.pop('city',None)
    return render_template("about.html", weather=weather_data)

@app.route('/select', methods=['POST','GET'])
def select():
    city = request.form.get('operator'); date = request.form.get('operator2')
    weather_data = None
    if 'weather_data' in session and 'weather_timestamp' in session:
        last = datetime.fromisoformat(session['weather_timestamp'])
        if datetime.now() - last < timedelta(minutes=3):
            weather_data = session['weather_data']
        else:
            session.pop('weather_data',None); session.pop('weather_timestamp',None); session.pop('city',None)
    city_data = df[df['City'] == city]
    city_year = city_data[city_data['Date'].astype(str).str.contains(date, na=False)].sort_values(by='Date', ascending=False)
    data_list = city_year.head(10).to_dict('records')
    return render_template("pollution.html", city=cities, data_list=data_list, new=f"{city}: Last 10 Entries (Year {date})", weather=weather_data)

@app.route('/assistant/chat', methods=['POST'])
def assistant_chat():
    data = request.get_json(force=True) or {}
    message = (data.get('message') or '').strip()
    sid = data.get('session_id') or session.get('assistant_sid')
    if not sid:
        sid = datetime.now().strftime('sess-%Y%m%d%H%M%S')
        session['assistant_sid'] = sid

    def reply(msg, options=None, link=None, ui_close=False):
        out = {"session_id": sid, "message": msg}
        if options: out["options"] = options
        if link: out["deep_link"] = {"url": link}
        if ui_close: out["ui"] = {"close": True}
        return jsonify(out)

    # Units/context
    units_symbol = "°F" if session.get('units','metric')=='imperial' else "°C"
    wind_unit = "mph" if session.get('units','metric')=='imperial' else "m/s"
    current_city = session.get('city') or session.get('default_city') or ''

    # Normalize and tokenize
    text_norm = normalize_text(message)
    tokens = tokenize(message)

    # Greetings
    if fuzzy_contains(tokens, ['hi','hello','hey','howdy','namaste','vanakkam','salaam','gm','good','morning','afternoon','evening']):
        return reply(
            "Hello! Ask about weather, UV, AQI, or open maps anytime.",
            options=["Weather now","UV now","AQI now","Open Outdoor map"]
        )

    # Exit/close
    if fuzzy_contains(tokens, ['bye','goodbye','cya','later','exit','quit','close','dismiss','stop','gn','night']):
        return reply("Okay, closing the assistant. See you soon!", ui_close=True)

    # Quick nav intents
    if fuzzy_contains(tokens, ['open','goto','launch']) and fuzzy_contains(tokens, ['agri','agriculture']):
        return reply("Opening Agriculture…", link="/agri_map")
    if fuzzy_contains(tokens, ['open','goto','launch']) and fuzzy_contains(tokens, ['outdoor','activity','activities']):
        return reply("Opening Outdoor map…", link="/outdoor_map")
    if fuzzy_contains(tokens, ['open','goto','launch']) and fuzzy_contains(tokens, ['stadium','venues','ground','grounds']):
        return reply("Opening Stadium map…", link="/stadium_map")

    # City extraction
    ask_city = extract_city_phrase(message) or extract_city_phrase(text_norm)
    target_city = ask_city or current_city

    # Word classification demo
    if fuzzy_contains(tokens, ['short','long']) and fuzzy_contains(tokens, ['word','words','token','tokens']):
        classes = classify_short_long_words(tokens)
        short_list = ', '.join(classes['short']) if classes['short'] else 'none'
        long_list = ', '.join(classes['long']) if classes['long'] else 'none'
        msg = f"Short words: {short_list} | Long words: {long_list}."
        return reply(msg, options=["Weather now", "UV now", "AQI now"])

    # UV
    if any_token(tokens, ['uv','uvi']) or fuzzy_contains(tokens, ['uv','uvi','sun','sunburn']):
        if not target_city:
            return reply("Which city for UV?", options=["Use current location","Thiruvananthapuram","Delhi"])
        w = get_weather_forecast(city=target_city)
        uvi_now = (w.get('uvi') or {}).get('now') if isinstance(w.get('uvi'), dict) else None
        level = (w.get('uvi') or {}).get('level') if isinstance(w.get('uvi'), dict) else None
        msg = f"{target_city}: UV now {uvi_now if uvi_now is not None else 'N/A'} ({level or 'N/A'})."
        return reply(msg, options=["Best time today","Open Outdoor map"])

    # AQI
    if any_token(tokens, ['aqi','air']) or fuzzy_contains(tokens, ['air','quality','pollution','smog']):
        if not target_city:
            return reply("Which city for AQI?", options=["Use current location","Thiruvananthapuram","Delhi"])
        aqi_val, aqi_bucket = get_aqi_for_city(target_city)
        if aqi_val is None:
            return reply(f"{target_city}: AQI data not available.", options=["Weather now","UV now"])
        return reply(f"{target_city}: AQI {int(aqi_val)} ({aqi_bucket}).", options=["Weather now","Best time today"])

    # Forecast/tomorrow
    if any_token(tokens, ['tomorrow','tmrw','forecast','later']) or fuzzy_contains(tokens, ['rain','precip','snow']):
        if not target_city:
            return reply("Which city for forecast?", options=["Use current location","Thiruvananthapuram","Delhi"])
        w = get_weather_forecast(city=target_city)
        fc = (w.get("forecast") or [])
        if fc:
            first = fc[0]
            msg = f"{target_city}: {first.get('date','')}: {first.get('description','N/A')} at {first.get('temp','N/A')}{units_symbol}."
            return reply(msg, options=["Weather now","UV now","AQI now"])
        return reply(f"{target_city}: No forecast available.", options=["Weather now","UV now"])

    # Best time
    if fuzzy_contains(tokens, ['best','good']) and fuzzy_contains(tokens, ['time','slot','when','train']):
        if not target_city:
            return reply("Which city should be used to find the best time?", options=["Use current location","Thiruvananthapuram","Delhi"])
        w = get_weather_forecast(city=target_city)
        desc = (w.get('description') or '').capitalize()
        temp = w.get('temp'); wind = w.get('wind_speed')
        uvi_now = (w.get('uvi') or {}).get('now') if isinstance(w.get('uvi'), dict) else None
        tip = "Early morning and late afternoon look most comfortable today."
        try:
            if uvi_now is not None and float(uvi_now) >= 8:
                tip = "Avoid midday due to high UV; prefer early morning or late evening."
        except Exception:
            pass
        if wind and wind >= 14:
            tip = "Windy conditions—consider a wind‑sheltered route or a later slot."
        msg = f"{target_city}: {desc}. Now {temp}{units_symbol}, wind {wind} {wind_unit}. {tip}"
        return reply(msg, options=["Open Outdoor map","Check tomorrow","Show rain chance"])

    # Weather now / today
    if fuzzy_contains(tokens, ['weather','wthr','now','today','temp','temperature','wind','humid','humidity']):
        if not target_city:
            return reply("Which city should the weather be checked for?", options=["Use current location","Thiruvananthapuram","Delhi"])
        w = get_weather_forecast(city=target_city)
        desc = (w.get('description') or '').capitalize()
        temp = w.get('temp'); rh = w.get('humidity'); wind = w.get('wind_speed')
        msg = f"{target_city}: {desc}. {temp}{units_symbol}. Humidity {rh}%. Wind {wind} {wind_unit}."
        return reply(msg, options=["Best time today","Open Agriculture","Open Outdoor map"])

    # Fallback
    return reply(
        "Try: “Weather in Kochi”, “UV now”, “AQI Delhi”, “Best time today”, or “Open Agriculture”.",
        options=["Weather now","UV now","AQI now","Open Outdoor map","Open Agriculture"]
    )



@app.route('/stadium_map', methods=['GET','POST'])
def stadium_map():
    search_query = request.form.get('search_query', '').lower() if request.method == 'POST' else ''
    filtered_stadiums = []; map_html=""; num_venues=0; stadium_data={}
    if search_query:
        filtered_stadiums = [v for v in stadiums if search_query in v['name'].lower() or search_query in v['city'].lower()]
        mp = pref('map', {'layer':'standard','cluster':True,'autocenter':'geo','reduced_motion':False})
        tiles = 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png' if mp.get('layer')=='standard' else 'https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png'
        attr = '© OpenStreetMap contributors'
        m = folium.Map(location=[20, 78], zoom_start=5, tiles=tiles, attr=attr, min_zoom=3, max_zoom=18)
        cluster=None
        if mp.get('cluster',True):
            try:
                from folium.plugins import MarkerCluster
                cluster = MarkerCluster().add_to(m)
            except Exception:
                cluster=None
        location_stadiums={}
        for venue in filtered_stadiums:
            key=(float(venue['lat']), float(venue['lon']))
            location_stadiums.setdefault(key, []).append(venue)
        for (lat,lon), venues_at_loc in location_stadiums.items():
            weather = get_weather_forecast(lat=lat, lon=lon, display_city=venues_at_loc[0]['city'])
            aqi_val, aqi_bucket = get_aqi_for_city(venues_at_loc[0]['city'])
            if weather is not None:
                uvi_now = weather.get("uvi",{}).get("now") if isinstance(weather.get("uvi"),dict) else None
                norm = dict(weather)
                norm["temp"]=weather.get("_norm",{}).get("temp_c", weather.get("temp",25))
                norm["wind_speed"]=weather.get("_norm",{}).get("wind_mps", weather.get("wind_speed",0))
                flags = pref('health_flags', [])
                adv = _build_advice(norm, aqi_val, uvi_now)
                if ('asthma' in flags or 'cardiac' in flags) and (aqi_val and aqi_val > 100):
                    if adv['outdoor'] != 'Avoid': adv['outdoor'] = 'Avoid'
                weather["advice"] = adv
            forecast = weather.get('forecast', [])[:4] if weather else []
            daily_forecast = weather.get('daily_forecast', [])[:2] if weather else []
            popup_html = "<div><strong>Stadiums at this location:</strong><br>" + "".join(
                f'<a href="#" class="sm-stadium-link" data-stadium="{v["name"]}">{v["name"]}</a><br>' for v in venues_at_loc
            ) + "</div>"
            target = cluster if cluster is not None else m
            folium.Marker([lat,lon], popup=folium.Popup(popup_html, max_width=300),
                          tooltip=venues_at_loc[0]['name'],
                          icon=folium.Icon(color='blue', icon='info-sign')).add_to(target)
            for v in venues_at_loc:
                stadium_data[v['name']] = {
                    'city': v['city'], 'lat': lat, 'lon': lon,
                    'temp': weather.get('temp','N/A') if weather else 'N/A',
                    'desc': weather.get('description','N/A') if weather else 'N/A',
                    'wind_speed': weather.get('wind_speed',0) if weather else 0,
                    'aqi_val': aqi_val, 'aqi_bucket': aqi_bucket,
                    'condition': get_playing_condition(weather or {}, aqi_bucket, v.get('sport','')),
                    'forecast': forecast or [], 'daily_forecast': daily_forecast or [],
                    'uvi': weather.get('uvi') if weather else {'now':None,'daily':[],'level':'N/A','color':'N/A'},
                    'advice': weather.get('advice') if weather else {'outdoor':'N/A','agri':'N/A','watering':'N/A','heat_index_c':'N/A'}
                }
        map_html = m._repr_html_()
        num_venues = len(filtered_stadiums)
    return render_template('stadium_map.html', map_html=map_html, num_venues=num_venues, stadium_data=stadium_data)

# New: Outdoor and Agriculture standalone pages
@app.route('/outdoor_map', methods=['GET','POST'])
def outdoor_map():
    posted_city = request.form.get('city', '').strip() if request.method == 'POST' else ''
    lat_raw = request.form.get('lat')
    lon_raw = request.form.get('lon')

    place_data = {}
    loc_precision = session.get('location_precision', 'precise')
    api_key = "31dc051843b6c8ba7f4a770a2b3237f8"

    center_lat, center_lon, do_center = 20.0, 78.0, False

    def add_place_record(label, w):
        if not w: return
        uvi_now = (w.get('uvi') or {}).get('now') if isinstance(w.get('uvi'), dict) else None
        norm = dict(w); norm['temp']=w.get('_norm',{}).get('temp_c', 25); norm['wind_speed']=w.get('_norm',{}).get('wind_mps', 0)
        adv = _build_advice(norm, None, uvi_now)
        place_data[label] = {
            "weather": {"temp": w['temp'], "wind_speed": w['wind_speed'], "description": w['description']},
            "uvi": w.get('uvi'),
            "advice": adv
        }

    # 1) Current location path
    if lat_raw and lon_raw:
        try:
            lat = float(lat_raw); lon = float(lon_raw)
            center_lat, center_lon, do_center = lat, lon, True

            if loc_precision == 'city':
                # Reverse geocode to city and fetch by name
                try:
                    geocode_url = f"http://api.openweathermap.org/geo/1.0/reverse?lat={lat}&lon={lon}&limit=1&appid={api_key}"
                    g = requests.get(geocode_url, timeout=8).json()
                    city_from_geo = g[0]['name'] if g and isinstance(g, list) and len(g) > 0 else None
                except Exception:
                    city_from_geo = None
                target_city = session.get('default_city') or city_from_geo or posted_city
                if target_city:
                    add_place_record(target_city, get_weather_forecast(city=target_city))
                else:
                    w = get_weather_forecast(lat=lat, lon=lon, display_city="Current location")
                    add_place_record("Current location", w)
            else:
                # Precise mode: fetch by coords
                label = posted_city or "Current location"
                w = get_weather_forecast(lat=lat, lon=lon, display_city=label)
                add_place_record(label, w)
        except Exception as e:
            print("outdoor_map: invalid geolocation input:", e)

    # 2) Optional city from form
    if posted_city:
        add_place_record(posted_city, get_weather_forecast(city=posted_city))

    # 3) Seed defaults if empty
    if not place_data:
        for seed in ['Thiruvananthapuram','Delhi']:
            add_place_record(seed, get_weather_forecast(city=seed))

    # Build map and center
    m = folium.Map(location=[center_lat, center_lon] if do_center else [20,78], zoom_start=11 if do_center else 5)
    map_html = m._repr_html_()
    return render_template('outdoor_map.html', map_html=map_html, place_data=place_data)

@app.route('/agri_map', methods=['GET','POST'])
def agri_map():
    # Inputs from form (city or geolocation)
    posted_city = request.form.get('city', '').strip() if request.method == 'POST' else ''
    lat_raw = request.form.get('lat')
    lon_raw = request.form.get('lon')

    place_data = {}
    agri_prefs = session.get('agri', {'crop':'','watering':'standard'})
    loc_precision = session.get('location_precision', 'precise')  # 'precise' or 'city'
    api_key = "31dc051843b6c8ba7f4a770a2b3237f8"  # same key used elsewhere

    # Decide map center (India default; overridden below)
    center_lat, center_lon, do_center = 20.0, 78.0, False

    def add_city_record(city_label, w):
        if not w:
            return
        uvi_now = (w.get('uvi') or {}).get('now') if isinstance(w.get('uvi'), dict) else None
        norm = dict(w)
        norm['temp'] = w.get('_norm', {}).get('temp_c', 25)
        norm['wind_speed'] = w.get('_norm', {}).get('wind_mps', 0)
        adv = _build_advice(norm, None, uvi_now)
        place_data[city_label] = {
            "weather": {"temp": w['temp'], "wind_speed": w['wind_speed'], "description": w['description']},
            "advice": adv,
            "agri": agri_prefs
        }

    # 1) If coordinates posted, use them according to privacy setting
    if lat_raw and lon_raw:
        try:
            lat = float(lat_raw); lon = float(lon_raw)
            # Center map at current location
            center_lat, center_lon, do_center = lat, lon, True

            if loc_precision == 'city':
                # Reverse geocode -> fetch by city
                try:
                    geocode_url = f"http://api.openweathermap.org/geo/1.0/reverse?lat={lat}&lon={lon}&limit=1&appid={api_key}"
                    g = requests.get(geocode_url, timeout=8).json()
                    city_from_geo = g[0]['name'] if g and isinstance(g, list) and len(g) > 0 else None
                except Exception:
                    city_from_geo = None
                target_city = session.get('default_city') or city_from_geo or posted_city
                if target_city:
                    add_city_record(target_city, get_weather_forecast(city=target_city))
                else:
                    # As a fallback still show precise point data but label it clearly
                    w = get_weather_forecast(lat=lat, lon=lon, display_city="Current location")
                    add_city_record("Current location", w)
            else:
                # Precise: fetch by coordinates and label as current location or posted city
                label = posted_city or "Current location"
                w = get_weather_forecast(lat=lat, lon=lon, display_city=label)
                add_city_record(label, w)
        except Exception as e:
            print("agri_map: invalid geolocation input:", e)

    # 2) If a city name was posted, include it as well
    if posted_city:
        add_city_record(posted_city, get_weather_forecast(city=posted_city))

    # 3) If nothing yet, seed with a couple of defaults
    if not place_data:
        for seed in ['Thiruvananthapuram','Delhi']:
            add_city_record(seed, get_weather_forecast(city=seed))

    # Build the map and center
    m = folium.Map(location=[center_lat, center_lon] if do_center else [20,78], zoom_start=11 if do_center else 5)
    map_html = m._repr_html_()
    return render_template('agri_map.html', map_html=map_html, place_data=place_data)


@app.route('/settings', methods=['GET','POST'])
def settings():
    supported = app.config.get('BABEL_SUPPORTED_LOCALES', ['en'])
    selected = session.get('lang', app.config.get('BABEL_DEFAULT_LOCALE','en'))
    if request.method == 'POST':
        lang  = request.form.get('lang'); units = request.form.get('units')
        if lang in supported: session['lang'] = lang
        if units in ['metric','imperial']: session['units'] = units
        tf = request.form.get('time_format'); dfmt = request.form.get('date_format')
        if tf in ['12h','24h']: session['time_format'] = tf
        if dfmt in ['DD-MM-YYYY','MM-DD-YYYY']: session['date_format'] = dfmt
        session['health_flags'] = request.form.getlist('health_flags')
        session['activities'] = {'sports': request.form.getlist('sports'), 'hours': request.form.get('training_hours') or ''}
        session['agri'] = {'crop': request.form.get('crop') or '', 'watering': request.form.get('watering_strategy') or 'standard'}
        session['map'] = {
            'layer': request.form.get('map_layer') or 'standard',
            'cluster': request.form.get('map_cluster') == 'on',
            'autocenter': request.form.get('autocenter') or 'geo',
            'reduced_motion': request.form.get('reduced_motion') == 'on'
        }
        session['location_precision'] = request.form.get('location_precision') or 'precise'
        session['default_city'] = request.form.get('default_city') or ''
        session['analytics'] = request.form.get('analytics') == 'on'
        session['alerts'] = request.form.getlist('alerts')
        session['quiet'] = {'start': request.form.get('quiet_start') or '', 'end': request.form.get('quiet_end') or ''}
        return redirect(request.referrer or url_for('settings'))
    current_units = session.get('units','metric')
    return render_template('settings.html',
        languages=supported, selected=selected, current_units=current_units,
        time_format=pref('time_format','24h'), date_format=pref('date_format','DD-MM-YYYY'),
        health_flags=pref('health_flags',[]),
        activities=pref('activities',{'sports':[],'hours':''}),
        agri=pref('agri',{'crop':'','watering':'standard'}),
        map_pref=pref('map',{'layer':'standard','cluster':True,'autocenter':'geo','reduced_motion':False}),
        location_precision=pref('location_precision','precise'),
        default_city=pref('default_city',''),
        analytics=pref('analytics',False),
        alerts=pref('alerts',[]),
        quiet=pref('quiet',{'start':'','end':''})
    )

if __name__ == "__main__":
    app.run(debug=True)
