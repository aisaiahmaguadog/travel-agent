"""
Japan Travel Agent - Powered by Claude AI
=========================================
Features: Flight search, Hotel search, Airbnb search, Chase Sapphire Preferred points comparison

Setup:
1. pip install anthropic requests apify-client
2. Set API keys in environment variables (see CONFIGURATION below)
3. Run: python agent.py
"""

import json
import math
import os
import time
from datetime import datetime
from typing import Any, Dict, List

import anthropic
import requests

# ============================================================
# CONFIGURATION
# ============================================================

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
AMADEUS_CLIENT_ID = os.getenv("AMADEUS_CLIENT_ID", "")
AMADEUS_CLIENT_SECRET = os.getenv("AMADEUS_CLIENT_SECRET", "")
RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY", "")
APIFY_TOKEN = os.getenv("APIFY_TOKEN", "")
APIFY_ACTOR_ID = os.getenv("APIFY_ACTOR_ID", "NDa1latMCCFtEgm6d")
APIFY_LOCATION_FIELD = os.getenv("APIFY_LOCATION_FIELD", "locationQuery")

MAX_TOOL_ITERATIONS_PER_TURN = 8

# Chase Sapphire Preferred = 1.25 cents per point through Chase Travel Portal
CHASE_CARD = "Chase Sapphire Preferred"
CHASE_CPP = 0.0125  # dollars per point value

AMADEUS_TOKEN_CACHE = {
    "access_token": None,
    "expires_at": 0,
}


def _parse_yyyy_mm_dd(date_str: str):
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except (TypeError, ValueError):
        return None


def _is_iata_code(value: str) -> bool:
    return isinstance(value, str) and len(value) == 3 and value.isalpha()


def _error(message: str, details=None) -> dict:
    payload = {"error": message}
    if details is not None:
        payload["details"] = details
    return payload


def _request_with_retries(method: str, url: str, **kwargs) -> requests.Response:
    last_exc = None

    for attempt in range(MAX_HTTP_RETRIES + 1):
        try:
            response = requests.request(method, url, timeout=REQUEST_TIMEOUT_SECONDS, **kwargs)
            if response.status_code in RETRYABLE_STATUS_CODES and attempt < MAX_HTTP_RETRIES:
                time.sleep(2 ** attempt)
                continue
            return response
        except requests.RequestException as exc:
            last_exc = exc
            if attempt < MAX_HTTP_RETRIES:
                time.sleep(2 ** attempt)
                continue
            raise

    raise requests.RequestException(str(last_exc) if last_exc else "Unknown HTTP error")


def _normalize_content_blocks(blocks: Any) -> List[Dict[str, Any]]:
    if isinstance(blocks, str):
        return [{"type": "text", "text": blocks}]

    normalized = []
    for block in blocks:
        if isinstance(block, dict):
            block_type = block.get("type")
            if block_type == "text":
                normalized.append({"type": "text", "text": block.get("text", "")})
            elif block_type == "tool_use":
                normalized.append(
                    {
                        "type": "tool_use",
                        "id": block.get("id"),
                        "name": block.get("name"),
                        "input": block.get("input", {}),
                    }
                )
            continue

        block_type = getattr(block, "type", None)
        if block_type == "text":
            normalized.append({"type": "text", "text": getattr(block, "text", "")})
        elif block_type == "tool_use":
            normalized.append(
                {
                    "type": "tool_use",
                    "id": getattr(block, "id", None),
                    "name": getattr(block, "name", None),
                    "input": getattr(block, "input", {}),
                }
            )

    return normalized


def _extract_text_blocks(blocks: Any) -> str:
    texts = []
    for block in blocks:
        if isinstance(block, dict):
            if block.get("type") == "text" and block.get("text"):
                texts.append(block["text"])
            continue

        if getattr(block, "type", None) == "text" and getattr(block, "text", None):
            texts.append(block.text)

    return "\n".join(texts).strip()


def _call_claude_with_retries(client: anthropic.Anthropic, **kwargs):
    last_exc = None
    for attempt in range(MAX_HTTP_RETRIES + 1):
        try:
            return client.messages.create(**kwargs)
        except Exception as exc:
            last_exc = exc
            if attempt < MAX_HTTP_RETRIES:
                time.sleep(2 ** attempt)
                continue
            raise

    raise RuntimeError(str(last_exc) if last_exc else "Unknown Claude API error")


# ============================================================
# FLIGHT SEARCH (Amadeus API)
# ============================================================


def get_amadeus_token():
    """Get a cached OAuth token from Amadeus when available."""
    if not AMADEUS_CLIENT_ID or not AMADEUS_CLIENT_SECRET:
        return _error("Missing Amadeus credentials. Set AMADEUS_CLIENT_ID and AMADEUS_CLIENT_SECRET.")

    now = time.time()
    cached_token = AMADEUS_TOKEN_CACHE.get("access_token")
    cached_exp = AMADEUS_TOKEN_CACHE.get("expires_at", 0)
    if cached_token and now < cached_exp - 60:
        return {"access_token": cached_token}

    try:
        resp = _request_with_retries(
            "POST",
            "https://test.api.amadeus.com/v1/security/oauth2/token",
            data={
                "grant_type": "client_credentials",
                "client_id": AMADEUS_CLIENT_ID,
                "client_secret": AMADEUS_CLIENT_SECRET,
            },
        )
        resp.raise_for_status()
        body = resp.json()
        token = body.get("access_token")
        expires_in = int(body.get("expires_in", 0))
        if not token:
            return _error("Amadeus token response missing access_token", resp.text)

        AMADEUS_TOKEN_CACHE["access_token"] = token
        AMADEUS_TOKEN_CACHE["expires_at"] = now + max(expires_in, 300)
        return {"access_token": token}
    except requests.RequestException as exc:
        return _error("Amadeus auth request failed", str(exc))
    except ValueError:
        return _error("Amadeus auth returned invalid JSON")


def search_flights(origin: str, destination: str, departure_date: str, return_date: str = None, adults: int = 1) -> dict:
    """
    Search for flights using Amadeus.
    origin/destination: IATA codes e.g. LAX, NRT, KIX, HND
    departure_date: YYYY-MM-DD
    """
    dep = _parse_yyyy_mm_dd(departure_date)
    ret = _parse_yyyy_mm_dd(return_date) if return_date else None
    if not _is_iata_code(origin) or not _is_iata_code(destination):
        return _error("origin and destination must be 3-letter IATA airport codes (e.g. LAX, HND).")
    if not dep:
        return _error("Invalid departure_date. Use YYYY-MM-DD.")
    if return_date and not ret:
        return _error("Invalid return_date. Use YYYY-MM-DD.")
    if dep and ret and ret <= dep:
        return _error("return_date must be after departure_date.")

    token_result = get_amadeus_token()
    token = token_result.get("access_token")
    if not token:
        return token_result

    params = {
        "originLocationCode": origin.upper(),
        "destinationLocationCode": destination.upper(),
        "departureDate": departure_date,
        "adults": adults,
        "max": 5,
        "currencyCode": "USD",
    }
    if return_date:
        params["returnDate"] = return_date

    headers = {"Authorization": f"Bearer {token}"}
    try:
        resp = _request_with_retries(
            "GET",
            "https://test.api.amadeus.com/v2/shopping/flight-offers",
            headers=headers,
            params=params,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as exc:
        return _error("Amadeus flight search failed", str(exc))
    except ValueError:
        return _error("Amadeus flight search returned invalid JSON")

    if "errors" in data:
        return _error("Amadeus returned errors", data["errors"])

    carriers = data.get("dictionaries", {}).get("carriers", {})
    results = []

    for offer in data.get("data", []):
        itineraries = offer.get("itineraries", [])
        if not itineraries:
            continue

        outbound = itineraries[0]
        out_segments = outbound.get("segments", [])
        if not out_segments:
            continue

        inbound = itineraries[1] if len(itineraries) > 1 else None
        in_segments = inbound.get("segments", []) if inbound else []

        airline_code = out_segments[0].get("carrierCode", "N/A")
        total_price = offer.get("price", {}).get("total")
        try:
            price_usd = float(total_price) if total_price is not None else None
        except (TypeError, ValueError):
            price_usd = None

        result = {
            "price_usd": price_usd,
            "airline": airline_code,
            "airline_name": carriers.get(airline_code, airline_code),
            "stops": len(out_segments) - 1,
            "duration": outbound.get("duration", "N/A"),
            "departure": out_segments[0].get("departure", {}).get("at", "N/A"),
            "arrival": out_segments[-1].get("arrival", {}).get("at", "N/A"),
        }

        if return_date:
            result["return_stops"] = len(in_segments) - 1 if in_segments else None
            result["return_duration"] = inbound.get("duration", "N/A") if inbound else None
            result["return_departure"] = in_segments[0].get("departure", {}).get("at", "N/A") if in_segments else None
            result["return_arrival"] = in_segments[-1].get("arrival", {}).get("at", "N/A") if in_segments else None

        results.append(result)

    return {
        "flights": results,
        "origin": origin.upper(),
        "destination": destination.upper(),
        "trip_type": "roundtrip" if return_date else "oneway",
    }


# ============================================================
# HOTEL SEARCH (RapidAPI — Hotels.com / Expedia backend)
# ============================================================


def _candidate_destination_ids(loc_data: Dict[str, Any]) -> List[str]:
    candidates = []
    for suggestion in loc_data.get("sr", []):
        if suggestion.get("type") not in {"CITY", "REGION"}:
            continue

        for key in ("gaiaId", "destinationId", "regionId"):
            value = suggestion.get(key)
            if value:
                as_str = str(value)
                if as_str not in candidates:
                    candidates.append(as_str)

    return candidates


def search_hotels(city: str, check_in: str, check_out: str, adults: int = 1) -> dict:
    """
    Search hotels via RapidAPI Hotels4 endpoint.
    city: plain text e.g. "Tokyo", "Kyoto", "Osaka"
    check_in / check_out: YYYY-MM-DD
    """
    in_date = _parse_yyyy_mm_dd(check_in)
    out_date = _parse_yyyy_mm_dd(check_out)
    if not in_date or not out_date:
        return _error("Invalid check-in/check-out date. Use YYYY-MM-DD.")
    if out_date <= in_date:
        return _error("check_out must be after check_in.")
    if not RAPIDAPI_KEY:
        return _error("Missing RAPIDAPI_KEY.")

    headers = {
        "X-RapidAPI-Key": RAPIDAPI_KEY,
        "X-RapidAPI-Host": "hotels4.p.rapidapi.com",
    }

    try:
        loc_resp = _request_with_retries(
            "GET",
            "https://hotels4.p.rapidapi.com/locations/v3/search",
            headers=headers,
            params={"q": city, "locale": "en_US"},
        )
        loc_resp.raise_for_status()
        loc_data = loc_resp.json()
    except requests.RequestException as exc:
        return _error("Hotels location lookup failed", str(exc))
    except ValueError:
        return _error("Hotels location lookup returned invalid JSON")

    destination_ids = _candidate_destination_ids(loc_data)
    if not destination_ids:
        return _error(f"Could not find city: {city}", loc_data)

    post_headers = {**headers, "Content-Type": "application/json"}
    selected_data = None
    selected_destination_id = None
    last_error = None

    for destination_id in destination_ids:
        payload = {
            "currency": "USD",
            "locale": "en_US",
            "siteId": 300000001,
            "destination": {"regionId": destination_id},
            "checkInDate": {
                "day": int(check_in.split("-")[2]),
                "month": int(check_in.split("-")[1]),
                "year": int(check_in.split("-")[0]),
            },
            "checkOutDate": {
                "day": int(check_out.split("-")[2]),
                "month": int(check_out.split("-")[1]),
                "year": int(check_out.split("-")[0]),
            },
            "rooms": [{"adults": adults}],
            "resultsStartingIndex": 0,
            "resultsSize": 8,
            "sort": "PRICE_LOW_TO_HIGH",
        }

        try:
            prop_resp = _request_with_retries(
                "POST",
                "https://hotels4.p.rapidapi.com/properties/v2/list",
                headers=post_headers,
                json=payload,
            )
            prop_resp.raise_for_status()
            prop_data = prop_resp.json()
            selected_data = prop_data
            selected_destination_id = destination_id

            properties = prop_data.get("data", {}).get("propertySearch", {}).get("properties", [])
            if properties:
                break
        except requests.HTTPError as exc:
            status = getattr(exc.response, "status_code", None)
            last_error = str(exc)
            if status in {400, 404}:
                continue
            return _error("Hotels property search failed", str(exc))
        except requests.RequestException as exc:
            return _error("Hotels property search failed", str(exc))
        except ValueError:
            return _error("Hotels property search returned invalid JSON")

    if selected_data is None:
        return _error("Hotels property search failed for all destination IDs", last_error or destination_ids)

    properties = selected_data.get("data", {}).get("propertySearch", {}).get("properties", [])
    hotels = []
    for prop in properties:
        raw_price = prop.get("price", {}).get("lead", {}).get("amount")
        try:
            nightly = round(float(raw_price), 2) if raw_price is not None else "N/A"
        except (TypeError, ValueError):
            nightly = "N/A"

        hotels.append(
            {
                "name": prop.get("name", "N/A"),
                "price_per_night": nightly,
                "rating": prop.get("reviews", {}).get("score", "N/A"),
                "review_count": prop.get("reviews", {}).get("total", 0),
                "price_note": "Price source may be nightly or stay-total depending on supplier response.",
            }
        )

    return {
        "hotels": hotels,
        "city": city,
        "check_in": check_in,
        "check_out": check_out,
        "destination_id_used": selected_destination_id,
    }


# ============================================================
# AIRBNB SEARCH (Apify scraper)
# ============================================================


def search_airbnbs(location: str, check_in: str, check_out: str, guests: int = 1) -> dict:
    """
    Search Airbnb via Apify actor.
    location: e.g. "Shinjuku, Tokyo, Japan"
    """
    in_date = _parse_yyyy_mm_dd(check_in)
    out_date = _parse_yyyy_mm_dd(check_out)
    if not in_date or not out_date:
        return _error("Invalid check-in/check-out date. Use YYYY-MM-DD.")
    if out_date <= in_date:
        return _error("check_out must be after check_in.")
    if not APIFY_TOKEN:
        return _error("Missing APIFY_TOKEN.")

    try:
        from apify_client import ApifyClient
    except ImportError:
        return _error("apify-client not installed. Run: pip install apify-client")

    client = ApifyClient(APIFY_TOKEN)
    run_input: Dict[str, Any] = {
        "checkIn": check_in,
        "checkOut": check_out,
        "adults": guests,
        "maxListings": 3,
        "currency": "USD",
    }

    if APIFY_LOCATION_FIELD == "locationQueries":
        run_input[APIFY_LOCATION_FIELD] = [location]
    else:
        run_input[APIFY_LOCATION_FIELD] = location

    extra_json = os.getenv("APIFY_EXTRA_INPUT_JSON", "")
    if extra_json:
        try:
            extra_input = json.loads(extra_json)
            if isinstance(extra_input, dict):
                run_input.update(extra_input)
        except json.JSONDecodeError:
            return _error("APIFY_EXTRA_INPUT_JSON is not valid JSON")

    try:
        run = client.actor(APIFY_ACTOR_ID).call(run_input=run_input)
        dataset_id = run.get("defaultDatasetId")
        if not dataset_id:
            return _error("Apify run did not return defaultDatasetId", run)

        listings = []
        for item in client.dataset(dataset_id).iterate_items():
            listings.append(
                {
                    "name": item.get("name"),
                    "price": item.get("price"),
                    "rating": item.get("rating"),
                    "room_type": item.get("roomType"),
                    "url": item.get("url"),
                    "price_note": "Price format varies by actor/listing; verify nightly vs total before booking.",
                }
            )

        return {
            "airbnbs": listings,
            "location": location,
            "actor_id_used": APIFY_ACTOR_ID,
            "location_field_used": APIFY_LOCATION_FIELD,
        }
    except Exception as exc:
        return _error("Apify Airbnb search failed", str(exc))


# ============================================================
# CHASE POINTS COMPARISON
# ============================================================


def compare_chase_points(cash_price: float, points_balance: int, category: str = "general") -> dict:
    """
    Compare paying cash vs using Chase Sapphire Preferred points.
    category: "flight" or "hotel" or "general"
    """
    if cash_price is None:
        return _error("cash_price is required")

    points_needed_portal = math.ceil(float(cash_price) / CHASE_CPP)
    points_after_portal = points_balance - points_needed_portal

    transfer_options = []
    if category == "flight":
        transfer_options = [
            {
                "partner": "ANA (via partner program)",
                "points_needed": 55000,
                "class": "Economy roundtrip to Japan",
                "typical_value": "$900-$1,200",
                "cpp": round(1050 / 55000 * 100, 2),
                "note": "Potential value only. Award availability and route rules vary.",
            },
            {
                "partner": "ANA Business Class (via partner program)",
                "points_needed": 60000,
                "class": "Business class roundtrip",
                "typical_value": "$3,000-$5,000",
                "cpp": round(4000 / 60000 * 100, 2),
                "note": "Potential value only. Verify partner award space before transferring.",
            },
            {
                "partner": "JAL (via partner program)",
                "points_needed": 60000,
                "class": "Economy roundtrip",
                "typical_value": "$900-$1,100",
                "cpp": round(1000 / 60000 * 100, 2),
                "note": "Potential value only. Partner charts and taxes can change.",
            },
        ]
    elif category == "hotel":
        transfer_options = [
            {
                "partner": "World of Hyatt",
                "points_needed": "Varies (8k-30k/night)",
                "class": "Hotel stay",
                "typical_value": "$150-$500/night",
                "cpp": "Up to 2.0+",
                "note": "Potential value only. Nights and categories vary by date and property.",
            }
        ]

    eligible_transfer_options = [
        t
        for t in transfer_options
        if isinstance(t.get("points_needed"), int)
        and isinstance(t.get("cpp"), float)
        and points_balance >= t["points_needed"]
    ]
    best_transfer = max(eligible_transfer_options, key=lambda t: t["cpp"]) if eligible_transfer_options else None

    if best_transfer and best_transfer["cpp"] > CHASE_CPP * 100:
        recommendation = (
            f"Potentially transfer to {best_transfer['partner']}. "
            f"Estimated value is ~{best_transfer['cpp']} cents/point vs portal {CHASE_CPP*100} cents/point, "
            "but confirm real award availability before transferring."
        )
    elif points_after_portal >= 0:
        recommendation = (
            f"Use Chase Travel Portal. You need {points_needed_portal:,} points "
            f"(you have {points_balance:,}) to cover this ${cash_price:.2f} purchase at 1.25 cents/point."
        )
    else:
        recommendation = (
            f"You don't have enough points ({points_balance:,}) to cover this "
            f"fully via portal (need {points_needed_portal:,}). Pay cash or use points partially."
        )

    return {
        "cash_price": cash_price,
        "your_points_balance": points_balance,
        "chase_card": CHASE_CARD,
        "portal_cpp": f"{CHASE_CPP*100} cents",
        "points_needed_portal": points_needed_portal,
        "points_after_portal": points_after_portal,
        "transfer_options": transfer_options,
        "recommendation": recommendation,
    }


# ============================================================
# CLAUDE AGENT SETUP
# ============================================================

tools = [
    {
        "name": "search_flights",
        "description": (
            "Search for real flights using Amadeus. Use IATA airport codes. "
            "Tokyo airports: NRT (Narita) or HND (Haneda). Osaka: KIX. "
            "Common US origins: LAX, JFK, SFO, ORD, SEA."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "origin": {"type": "string", "description": "IATA code e.g. LAX"},
                "destination": {"type": "string", "description": "IATA code e.g. NRT"},
                "departure_date": {"type": "string", "description": "YYYY-MM-DD"},
                "return_date": {"type": "string", "description": "YYYY-MM-DD (optional)"},
                "adults": {"type": "integer", "description": "Number of passengers"},
            },
            "required": ["origin", "destination", "departure_date"],
        },
    },
    {
        "name": "search_hotels",
        "description": "Search for hotels in a Japanese city with real pricing.",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "e.g. Tokyo, Kyoto, Osaka"},
                "check_in": {"type": "string", "description": "YYYY-MM-DD"},
                "check_out": {"type": "string", "description": "YYYY-MM-DD"},
                "adults": {"type": "integer"},
            },
            "required": ["city", "check_in", "check_out"],
        },
    },

    {
        "name": "compare_chase_points",
        "description": (
            "Compare paying cash vs using Chase Sapphire Preferred points for a purchase. "
            "Always call this after finding flight or hotel prices so the user can decide."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "cash_price": {"type": "number", "description": "Price in USD"},
                "points_balance": {"type": "integer", "description": "User's current Chase points"},
                "category": {"type": "string", "enum": ["flight", "hotel", "general"]},
            },
            "required": ["cash_price", "points_balance"],
        },
    },
]

SYSTEM_PROMPT = """You are Kenji, an expert Japan travel agent and Chase points optimizer.
You help users plan incredible 2-week Japan trips by searching real flight prices, hotels,
and Airbnbs - and always comparing whether Chase Sapphire Preferred points give better value.

Your personality: friendly, knowledgeable, proactive. You love Japan and want the user to
get the most out of their trip and their points.

At the start of every conversation, ask the user for:
1. Their departure city/airport
2. Travel dates (or approximate dates)
3. Their Chase points balance
4. Budget range (flights + accommodation)
5. Interests (food, temples, anime, nature, nightlife, etc.)

Then:
- Search flights and show top 3 options with prices
- ALWAYS run compare_chase_points after finding flight prices
- Search hotels AND Airbnbs for each city they plan to visit
- Run compare_chase_points for hotel costs too
- Recommend the best payment method for each purchase
- Build a day-by-day 2-week Japan itinerary around their bookings

Japan knowledge to apply:
- Classic 2-week route: Tokyo (5 days) -> Hakone/Nikko (2 days) -> Kyoto (3 days) -> Osaka (2 days) -> Hiroshima/Nara day trips
- JR Pass may or may not be worth it after fare changes; compare pass cost against your exact long-distance routes
- Get a Suica IC card for local trains
- Book popular restaurants (like ramen shops) in advance
- Cherry blossom season (late March-April) and fall foliage (Nov) are peak - book early

Always show actual numbers from search results. Never make up prices."""


def _ensure_required_env() -> bool:
    missing = []
    if not ANTHROPIC_API_KEY:
        missing.append("ANTHROPIC_API_KEY")
    if missing:
        print(f"Missing required environment variables: {', '.join(missing)}")
        return False
    return True


def run_agent():
    if not _ensure_required_env():
        return

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    history: List[Dict[str, Any]] = []

    print("=" * 60)
    print("JAPAN TRAVEL AGENT - Powered by Claude AI")
    print("=" * 60)
    print("Type your message and press Enter. Type 'quit' to exit.\n")

    try:
        opening = _call_claude_with_retries(
            client,
            model=ANTHROPIC_MODEL,
            max_tokens=512,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": "Hello, I want to plan a 2-week Japan trip."}],
        )
        opening_text = _extract_text_blocks(opening.content)
        print(f"Kenji: {opening_text}\n")
        history.append({"role": "user", "content": "Hello, I want to plan a 2-week Japan trip."})
        history.append({"role": "assistant", "content": _normalize_content_blocks(opening.content)})
    except Exception as exc:
        print(f"Failed to start conversation: {exc}")
        return

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("quit", "exit", "bye"):
            print("\nKenji: Safe travels! Enjoy Japan!")
            break
        if not user_input:
            continue

        history.append({"role": "user", "content": user_input})
        tool_iteration_count = 0

        while True:
            if tool_iteration_count >= MAX_TOOL_ITERATIONS_PER_TURN:
                msg = (
                    "I reached the tool-call safety limit for this turn. "
                    "Please refine your request (for example: fewer cities or one search type at a time)."
                )
                print(f"\nKenji: {msg}\n")
                history.append({"role": "assistant", "content": [{"type": "text", "text": msg}]})
                break

            try:
                response = _call_claude_with_retries(
                    client,
                    model=ANTHROPIC_MODEL,
                    max_tokens=4096,
                    system=SYSTEM_PROMPT,
                    tools=tools,
                    messages=history,
                )
            except Exception as exc:
                print(f"\nKenji: I hit an API error while planning your trip: {exc}\n")
                break

            normalized_response_content = _normalize_content_blocks(response.content)

            if response.stop_reason == "end_turn":
                reply = _extract_text_blocks(response.content)
                print(f"\nKenji: {reply}\n")
                history.append({"role": "assistant", "content": normalized_response_content})
                break

            if response.stop_reason == "tool_use":
                tool_iteration_count += 1
                history.append({"role": "assistant", "content": normalized_response_content})
                tool_results = []

                for block in response.content:
                    if getattr(block, "type", "") != "tool_use":
                        continue

                    print(f"  Searching {block.name.replace('_', ' ')}...")
                    try:
                        if block.name == "search_flights":
                            result = search_flights(**block.input)
                        elif block.name == "search_hotels":
                            result = search_hotels(**block.input)
                        elif block.name == "search_airbnbs":
                            result = search_airbnbs(**block.input)
                        elif block.name == "compare_chase_points":
                            result = compare_chase_points(**block.input)
                        else:
                            result = _error(f"Unknown tool: {block.name}")
                    except Exception as exc:
                        result = _error(f"Tool {block.name} crashed", str(exc))

                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": json.dumps(result),
                        }
                    )

                history.append({"role": "user", "content": tool_results})
                continue

            history.append({"role": "assistant", "content": normalized_response_content})
            print(
                f"\nKenji: I stopped at state '{response.stop_reason}'. "
                "Please retry or refine your request.\n"
            )
            break


if __name__ == "__main__":
    run_agent()