from flask import Flask, render_template, request, jsonify
import os
import requests
import json
import datetime
import math
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# 1. Get the absolute path of the directory containing this file (the 'api' folder)
basedir = os.path.abspath(os.path.dirname(__file__))

# 2. Go up one level to the root directory (the 'Team Nova' folder)
root_dir = os.path.dirname(basedir)

# 3. Join the root directory with the 'templates' folder
template_dir = os.path.join(root_dir, 'templates')

# 4. Initialize Flask using the absolute path to the root templates folder
app = Flask(__name__, template_folder=template_dir)

# Initialize Groq Client
# Ensure GROQ_API_KEY is securely set in your .env or Vercel Environment Variables
groq_api_key = os.environ.get("GROQ_API_KEY", "")
client = Groq(api_key=groq_api_key)

def _fetch_public_holidays(target_date=None):
    """Real upcoming Vietnamese public holidays within 14 days of target_date, via Nager.Date (free, keyless).

    Returns:
        list: holidays inside the window (possibly empty) when at least one year's
              API call succeeded — an empty list is a confirmed "no holidays" result.
        None: when EVERY queried year's API call failed, so we genuinely do not know.
    """
    if target_date is None:
        target_date = datetime.date.today()
    window_end = target_date + datetime.timedelta(days=14)

    holidays = []
    any_success = False
    for year in {target_date.year, window_end.year}:
        url = f"https://date.nager.at/api/v3/PublicHolidays/{year}/VN"
        try:
            response = requests.get(url, timeout=5)
            if response.status_code != 200:
                continue
            payload = response.json()
            year_holidays = []
            for item in payload:
                holiday_date = datetime.date.fromisoformat(item["date"])
                if target_date <= holiday_date <= window_end:
                    year_holidays.append({"date": item["date"], "name": item.get("localName") or item.get("name")})
        except (requests.exceptions.RequestException, ValueError, KeyError, TypeError, AttributeError):
            continue
        any_success = True
        holidays.extend(year_holidays)

    if not any_success:
        return None
    return holidays


def _fetch_egg_price_benchmark_usd_per_dozen():
    """Real US egg price (USD/dozen), via USDA NASS Quick Stats (free, requires signup key)."""
    api_key = os.environ.get("USDA_API_KEY", "")
    url = "https://quickstats.nass.usda.gov/api/api_GET/"
    params = {
        "key": api_key,
        "commodity_desc": "EGGS",
        "statisticcat_desc": "PRICE RECEIVED",
        "agg_level_desc": "NATIONAL",
        "format": "JSON",
    }
    try:
        response = requests.get(url, params=params, timeout=5)
        if response.status_code != 200:
            return None
        records = response.json().get("data", [])

        # Filter records defensively: skip any with non-string or missing unit_desc/value
        dozen_records = []
        for r in records:
            unit_desc = r.get("unit_desc", "")
            # USDA NASS Quick Stats returns the field as "Value" (capital V).
            value = r.get("Value", r.get("value", ""))
            # Only process if both fields are strings and value is numeric
            if isinstance(unit_desc, str) and isinstance(value, str):
                if "DOZEN" in unit_desc.upper() and value.replace(".", "", 1).isdigit():
                    dozen_records.append(r)

        if not dozen_records:
            return None

        latest = max(dozen_records, key=lambda r: r.get("year", "0"))
        latest_value = latest.get("Value", latest.get("value"))
        return {"price_usd_per_dozen": float(latest_value), "year": latest["year"]}
    except (requests.exceptions.RequestException, ValueError, KeyError, TypeError, AttributeError):
        return None


def _fetch_usd_to_vnd_rate():
    """Real live USD->VND exchange rate, via open.er-api.com (free, keyless)."""
    try:
        response = requests.get("https://open.er-api.com/v6/latest/USD", timeout=5)
        if response.status_code != 200:
            return None
        data = response.json()
        return data.get("rates", {}).get("VND")
    except (requests.exceptions.RequestException, ValueError, KeyError, TypeError, AttributeError):
        return None

_OVERPASS_CATEGORY_TAGS = {
    "Mini-supermarket chains": [("shop", "supermarket"), ("shop", "convenience")],
    "Clean food stores": [("shop", "organic"), ("shop", "greengrocer")],
    "Industrial Kitchens & Canteens": [("amenity", "canteen")],
    "High-end Restaurants": [("amenity", "restaurant")],
}

# `requests`' own `timeout=` only bounds gaps between chunks, not total wall-clock time — a
# server trickling data slowly can still take 60s+ even with a 15s timeout. Running the call
# on a worker thread and enforcing a hard deadline via future.result() is what actually caps
# how long an interactive scan waits.
_OVERPASS_EXECUTOR = ThreadPoolExecutor(max_workers=4)


def _post_with_hard_timeout(url, data, headers, request_timeout, wall_clock_timeout):
    future = _OVERPASS_EXECUTOR.submit(requests.post, url, data=data, headers=headers, timeout=request_timeout)
    return future.result(timeout=wall_clock_timeout)


def _haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    return r * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


_OVERPASS_CACHE = {}
_OVERPASS_CACHE_TTL_SECONDS = 600


def _query_overpass_businesses(lat, lon, radius_km, category):
    """Real business listings from OpenStreetMap via Overpass. Returns [] on any failure — never raises.

    The public Overpass instance can take 30-80s under load, which is too slow for an
    interactive scan. We cap the wait so a slow instance fails fast (honest empty result,
    same as any other unavailable real-data source) instead of hanging the request, and we
    cache successful lookups briefly so repeat scans of the same area are instant.
    """
    tags = _OVERPASS_CATEGORY_TAGS.get(category, [("shop", "supermarket")])

    try:
        # Coerce here too (defence in depth): the callers validate, but the helper's
        # contract is to return [] rather than raise on any bad input or response.
        radius_m = int(float(radius_km) * 1000)
        safe_lat, safe_lon = float(lat), float(lon)
    except (TypeError, ValueError):
        return []

    cache_key = (round(safe_lat, 3), round(safe_lon, 3), radius_m, category)
    cached = _OVERPASS_CACHE.get(cache_key)
    if cached is not None and (time.time() - cached[0]) < _OVERPASS_CACHE_TTL_SECONDS:
        return cached[1]

    results = []
    try:
        # `nwr` matches nodes, ways AND relations — many real supermarkets/restaurants
        # are mapped as building ways or multipolygon relations, not bare nodes.
        clauses = "\n".join(
            f'  nwr["{key}"="{value}"](around:{radius_m},{safe_lat},{safe_lon});' for key, value in tags
        )
        query = f"[out:json][timeout:55];\n(\n{clauses}\n);\nout center 50;"

        # Overpass rejects requests without a real User-Agent (406 Not Acceptable). Correctness
        # (finding the real businesses that are actually there) matters more than speed here, so
        # the wall-clock deadline is generous (90s) rather than cutting the public instance off
        # before it can finish under load — repeat scans of the same area are instant anyway via
        # the cache above.
        response = _post_with_hard_timeout(
            "https://overpass-api.de/api/interpreter",
            data={"data": query},
            headers={"User-Agent": os.environ.get("NOMINATIM_USER_AGENT", "TeamNovaEggDistro/1.0")},
            request_timeout=90,
            wall_clock_timeout=90,
        )
        if response.status_code != 200:
            return []
        elements = response.json().get("elements", [])

        for el in elements:
            el_tags = el.get("tags") or {}
            if not isinstance(el_tags, dict):
                continue
            name = el_tags.get("name")
            if not name:
                continue
            center = el.get("center") or {}
            if not isinstance(center, dict):
                center = {}
            el_lat = el.get("lat") if el.get("lat") is not None else center.get("lat")
            el_lon = el.get("lon") if el.get("lon") is not None else center.get("lon")
            if el_lat is None or el_lon is None:
                continue
            try:
                # Downstream scoring does float maths on these — skip an element with
                # unusable coordinates rather than dropping the whole result set.
                el_lat, el_lon = float(el_lat), float(el_lon)
            except (TypeError, ValueError):
                continue
            category_tag = el_tags.get("shop") or el_tags.get("amenity") or "unknown"
            results.append({"name": name, "lat": el_lat, "lon": el_lon, "osm_category": category_tag})
    except (requests.exceptions.RequestException, FutureTimeoutError, ValueError, KeyError, TypeError, AttributeError):
        return []

    _OVERPASS_CACHE[cache_key] = (time.time(), results)
    return results


def _score_leads(businesses, depot_lat, depot_lon, radius_km):
    scored = []
    for business in businesses:
        distance_km = round(_haversine_km(depot_lat, depot_lon, business["lat"], business["lon"]), 2)
        proximity_score = max(0.0, 100.0 - (distance_km / max(radius_km, 0.1)) * 100.0)
        same_category_nearby = sum(
            1 for other in businesses
            if other["osm_category"] == business["osm_category"]
            and _haversine_km(business["lat"], business["lon"], other["lat"], other["lon"]) <= 1.0
        )
        density_score = min(100.0, same_category_nearby * 10.0)
        lead_score = round(0.6 * proximity_score + 0.4 * density_score)

        if lead_score >= 70:
            priority = "High"
        elif lead_score >= 40:
            priority = "Medium"
        else:
            priority = "Low"

        scored.append({**business, "distance_km": distance_km, "lead_score": lead_score, "contact_priority": priority})

    return sorted(scored, key=lambda b: b["lead_score"], reverse=True)


def _geocode_address(address, user_agent):
    try:
        response = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": address, "format": "json", "limit": 1},
            headers={"User-Agent": user_agent},
            timeout=5,
        )
        if response.status_code != 200:
            return None
        results = response.json()
        if not results:
            return None
        return {"lat": float(results[0]["lat"]), "lon": float(results[0]["lon"])}
    except (requests.exceptions.RequestException, ValueError, KeyError, TypeError, AttributeError, IndexError):
        return None


def _get_distance_duration_matrix(coords, api_key):
    try:
        response = requests.post(
            "https://api.openrouteservice.org/v2/matrix/driving-car",
            json={"locations": [[lon, lat] for lon, lat in coords], "metrics": ["distance", "duration"], "units": "km"},
            headers={"Authorization": api_key, "Content-Type": "application/json"},
            timeout=5,
        )
        if response.status_code != 200:
            return None
        data = response.json()
        return {"distances": data["distances"], "durations": data["durations"]}
    except (requests.exceptions.RequestException, ValueError, KeyError, TypeError, AttributeError):
        return None


def _solve_route_sequence(depot, deliveries, matrix, dispatch_time_str):
    distances = matrix["distances"]
    durations = matrix["durations"]
    n = len(deliveries)

    unvisited = set(range(1, n + 1))  # matrix indices 1..n correspond to deliveries[0..n-1]
    current_index = 0
    current_time = datetime.datetime.strptime(dispatch_time_str, "%H:%M")
    total_km = 0.0
    route = []
    step_num = 1

    while unvisited:
        next_index = min(unvisited, key=lambda i: durations[current_index][i])
        travel_seconds = durations[current_index][next_index]
        travel_km = distances[current_index][next_index]
        arrival_time = current_time + datetime.timedelta(seconds=travel_seconds)

        delivery = deliveries[next_index - 1]
        window_start_str, window_end_str = delivery["time_window"].split("-")
        window_start = datetime.datetime.strptime(window_start_str, "%H:%M").replace(
            year=arrival_time.year, month=arrival_time.month, day=arrival_time.day
        )
        window_end = datetime.datetime.strptime(window_end_str, "%H:%M").replace(
            year=arrival_time.year, month=arrival_time.month, day=arrival_time.day
        )

        # Handle time windows crossing midnight (e.g., "22:00-02:00")
        if window_end < window_start:
            # Overnight window. Decide which day-boundary shift applies to this arrival.
            if arrival_time <= window_end:
                # Arrival is in the early-morning portion — the window that applies opened yesterday.
                window_start -= datetime.timedelta(days=1)
            else:
                # Arrival is in the evening portion (or before the window opens) — window closes tomorrow.
                window_end += datetime.timedelta(days=1)

        if arrival_time < window_start:
            status = "Early"
        elif arrival_time > window_end:
            status = "Risk of Delay"
        else:
            status = "Met"

        route.append({
            "step": step_num,
            "name": delivery["name"],
            "arrival_time": arrival_time.strftime("%H:%M"),
            "time_window_status": status,
        })

        total_km += travel_km
        current_time = max(arrival_time, window_start)
        current_index = next_index
        unvisited.remove(next_index)
        step_num += 1

    return_seconds = durations[current_index][0]
    return_km = distances[current_index][0]
    return_arrival = current_time + datetime.timedelta(seconds=return_seconds)
    route.append({
        "step": step_num,
        "name": depot["name"],
        "arrival_time": return_arrival.strftime("%H:%M"),
        "time_window_status": "Met",
    })
    total_km += return_km

    return {"total_estimated_km": round(total_km, 2), "optimized_route": route}


@app.route('/')
def home():
    """Renders the landing page."""
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    """Renders the main functional dashboard."""
    return render_template('dashboard.html')

@app.route('/market-tracker')
def market_tracker():
    """Renders the Global Market Benchmark module."""
    return render_template('market_tracker.html')

@app.route('/distributor-map')
def distributor_map():
    """Renders the Strategic Distributor Map module."""
    return render_template('distributor_map.html')

@app.route('/route-optimizer')
def route_optimizer():
    """Renders the Smart Route & Logistics Optimizer module."""
    return render_template('route_optimizer.html')

@app.route('/api/predict_demand', methods=['POST'])
def predict_demand():
    """
    Module 1: AI Demand & Inventory Predictor
    Calculates inventory adjustments based on real-time weather and events.
    """
    try:
        data = request.json
        product = data.get('product', 'Egg Cartons')
        baseline = data.get('baseline', 100)
        events = data.get('events', 'None')
        
        # Hardcoded coordinates for Hanoi based on the system blueprint examples
        lat, lon = 21.0285, 105.8542 
        
        # 1. Fetch Real-World Weather Context (Open-Meteo Free API)
        weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
        weather_response = requests.get(weather_url, timeout=5)

        weather_context = "Unknown Weather Conditions"
        if weather_response.status_code == 200:
            cw = weather_response.json().get('current_weather', {})
            weather_context = f"Temperature: {cw.get('temperature')}°C, Windspeed: {cw.get('windspeed')} km/h, WeatherCode: {cw.get('weathercode')}"

        real_holidays = _fetch_public_holidays()
        if real_holidays is None:
            # The holiday source was unreachable — do NOT claim there are zero holidays.
            holiday_context = "Holiday data temporarily unavailable"
        elif real_holidays:
            holiday_context = "; ".join(f"{h['name']} on {h['date']}" for h in real_holidays)
        else:
            holiday_context = "No public holidays in the next 14 days"

        # 2. Formulate Prompt for Groq Engine
        system_prompt = (
            "You are an AI Supply Chain Analyst. You output ONLY strictly valid JSON. "
            "Do not include Markdown blocks, preambles, or formatting tags like ```json. "
            "Output the raw JSON object only."
        )
        
        user_prompt = f"""
        Analyze the following demand factors and predict the required upcoming inventory to prevent overstocking and understocking.

        Target Product: {product}
        Location Context: Hanoi, Vietnam
        Baseline Daily Demand: {baseline} units
        Current Weather Forecast: {weather_context}
        Upcoming Events/Seasonality (user-reported): {events}
        Confirmed Real Public Holidays (next 14 days, Vietnam): {holiday_context}

        Calculate the new target demand volume.
        Return a JSON object with EXACTLY these keys:
        {{
            "recommended_production": <integer value calculated>,
            "weather_impact": "<string analyzing how the weather specifically alters foot traffic/demand>",
            "event_impact": "<string analyzing how the event/seasonality scales the demand>",
            "reasoning": "<string providing the final strategic summary for the production team>"
        }}
        """

        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model="openai/gpt-oss-120b", 
            temperature=0.2
        )

        result = chat_completion.choices[0].message.content
        result = result.replace('```json', '').replace('```', '').strip()

        parsed = json.loads(result)
        parsed["detected_holidays"] = real_holidays
        return jsonify(parsed)
        
    except json.JSONDecodeError:
        return jsonify({"error": "Failed to parse AI response into valid JSON."}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/track_prices', methods=['POST'])
def track_prices():
    """
    Module 2: Market Price Benchmark
    Compares the user's price against a real, live global egg price benchmark (USDA + live forex).
    """
    try:
        data = request.json
        product = data.get('product', 'Clean Indigenous Chicken Eggs')
        my_price_per_dozen = float(data.get('my_price', 0))

        benchmark = _fetch_egg_price_benchmark_usd_per_dozen()
        fx_rate = _fetch_usd_to_vnd_rate()

        if benchmark is None or fx_rate is None:
            return jsonify({
                "error": "Real market benchmark data is temporarily unavailable (USDA or exchange rate source did not respond). Please try again shortly."
            }), 503

        benchmark_price_vnd = round(benchmark["price_usd_per_dozen"] * fx_rate, 0)

        system_prompt = (
            "You are an AI Market Intelligence Analyst. You output ONLY strictly valid JSON. "
            "Do not include Markdown blocks, preambles, or formatting tags like ```json. "
            "Output the raw JSON object only."
        )

        user_prompt = f"""
        Compare the user's wholesale price against a real global benchmark for {product}.
        User's Current Price: {my_price_per_dozen} VND per dozen
        Real Global Benchmark: {benchmark_price_vnd} VND per dozen (source: USDA NASS, {benchmark['year']}, converted at live USD/VND rate {fx_rate})

        Note this benchmark reflects US wholesale market conditions, not local Vietnamese retail competitors.
        Identify whether the user's price is below, near, or above this global benchmark, and give brief
        strategic context for what that might mean locally.

        Return a JSON object with EXACTLY these keys:
        {{
            "competitive_status": "<string, e.g., 'Below Global Benchmark', 'Near Global Benchmark', 'Above Global Benchmark'>",
            "actionable_insight": "<string, 2-3 sentences of direct strategic advice for the business manager>"
        }}
        """

        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model="openai/gpt-oss-120b",
            temperature=0.1
        )

        result = chat_completion.choices[0].message.content
        result = result.replace('```json', '').replace('```', '').strip()
        ai_analysis = json.loads(result)

        return jsonify({
            "benchmark": {
                "price_vnd_per_dozen": benchmark_price_vnd,
                "source": "USDA NASS Quick Stats",
                "source_year": benchmark["year"],
                "fx_rate_usd_vnd": fx_rate
            },
            "your_price_vnd_per_dozen": my_price_per_dozen,
            "analysis": ai_analysis
        })

    except json.JSONDecodeError:
        return jsonify({"error": "Failed to parse AI response into valid JSON."}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/scan_distributors', methods=['POST'])
def scan_distributors():
    """
    Module 3: Strategic Distributor Map
    Real B2B leads from OpenStreetMap, scored by real proximity/density signals.
    """
    try:
        data = request.json or {}

        # Coerce and validate before ANY value reaches the Overpass QL string builder.
        # Un-coerced values would be interpolated verbatim (query-injection risk), and a
        # string radius would be repeated by `radius * 1000` rather than multiplied.
        try:
            lat = float(data.get('lat', 21.0285))
            lon = float(data.get('lon', 105.8542))
            radius = max(1, min(50, float(data.get('radius', 5))))
        except (TypeError, ValueError):
            return jsonify({"error": "lat, lon, and radius must be valid numbers"}), 400

        category = data.get('category', 'Mini-supermarket chains')

        businesses = _query_overpass_businesses(lat, lon, radius, category)
        if not businesses:
            return jsonify({"status": "success", "data": [], "note": "No real listings found in this radius via OpenStreetMap."})

        scored = _score_leads(businesses, lat, lon, radius)[:15]

        summary_note = ""
        try:
            system_prompt = (
                "You are a B2B sales strategist. Summarize the following REAL lead list in 1-2 sentences. "
                "Do not invent any businesses or numbers beyond what is given. Output plain text only."
            )
            top_names = ", ".join(b["name"] for b in scored[:5])
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Top real leads found near the target area: {top_names}. Category: {category}."}
                ],
                model="openai/gpt-oss-120b",
                temperature=0.2,
                timeout=10
            )
            summary_note = chat_completion.choices[0].message.content.strip()
        except Exception:
            summary_note = ""

        return jsonify({"status": "success", "data": scored, "summary": summary_note})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/optimize_route', methods=['POST'])
def optimize_route():
    """
    Module 4: Smart Route & Logistics Optimizer
    Real geocoding (Nominatim) + real road-network routing (OpenRouteService) + deterministic sequencing.
    """
    try:
        data = request.json
        depot_input = data.get('depot', {"name": "Central Warehouse", "lat": 21.0285, "lon": 105.8542})
        deliveries_input = data.get('deliveries', [])
        dispatch_time = data.get('dispatch_time', '05:30')

        if not deliveries_input:
            return jsonify({"error": "At least one delivery address is required."}), 400
        if len(deliveries_input) > 8:
            return jsonify({"error": "A maximum of 8 delivery stops is supported per request."}), 400

        user_agent = os.environ.get("NOMINATIM_USER_AGENT", "TeamNovaEggDistro/1.0")
        ors_key = os.environ.get("ORS_API_KEY", "")

        geocode_called = False

        if "lat" in depot_input and "lon" in depot_input:
            depot_coords = {"lat": float(depot_input["lat"]), "lon": float(depot_input["lon"])}
        else:
            depot_coords = _geocode_address(depot_input.get("address", depot_input.get("name", "")), user_agent)
            geocode_called = True
            if depot_coords is None:
                return jsonify({"error": f"Could not geocode depot location."}), 422

        deliveries = []
        for i, item in enumerate(deliveries_input):
            if geocode_called or i > 0:
                time.sleep(1.1)  # respect Nominatim's 1 request/second policy
            coords = _geocode_address(item["address"], user_agent)
            geocode_called = True
            if coords is None:
                return jsonify({"error": f"Could not geocode address: {item['address']}"}), 422
            deliveries.append({"name": item["name"], "time_window": item["time_window"], **coords})

        all_coords = [(depot_coords["lon"], depot_coords["lat"])] + [(d["lon"], d["lat"]) for d in deliveries]
        matrix = _get_distance_duration_matrix(all_coords, ors_key)
        if matrix is None:
            return jsonify({"error": "Real routing data is temporarily unavailable (OpenRouteService did not respond)."}), 503

        route_result = _solve_route_sequence({"name": depot_input.get("name", "Central Warehouse")}, deliveries, matrix, dispatch_time)

        strategy_summary = ""
        try:
            system_prompt = (
                "You are a logistics analyst. In 1-2 sentences, summarize the REAL computed route below. "
                "Do not invent any distances, times, or stops beyond what is given. Output plain text only."
            )
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": json.dumps(route_result)}
                ],
                model="openai/gpt-oss-120b",
                temperature=0.1,
                timeout=10
            )
            strategy_summary = chat_completion.choices[0].message.content.strip()
        except Exception:
            strategy_summary = ""

        route_result["strategy_summary"] = strategy_summary

        # Build name-keyed coordinate lookup to handle reordered stops correctly
        coords_by_name = {d["name"]: d for d in deliveries}
        coords_by_name[depot_input.get("name", "Central Warehouse")] = depot_coords

        for step in route_result["optimized_route"]:
            coords_source = coords_by_name[step["name"]]
            step["lat"] = coords_source["lat"]
            step["lon"] = coords_source["lon"]

        return jsonify({"status": "success", "data": route_result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)