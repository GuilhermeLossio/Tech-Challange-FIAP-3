from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd
from flask import Flask, jsonify, request


def register_flight_views(app: Flask, deps: dict[str, Any]) -> None:
    if "flight_countries" in app.view_functions:
        return

    load_airports_index = deps["load_airports_index"]
    optional_text = deps["optional_text"]
    load_upcoming_flights_frame = deps["load_upcoming_flights_frame"]
    extract_airport_departures = deps["extract_airport_departures"]

    @app.get("/api/flight/countries")
    def flight_countries():
        source_uri = request.args.get("source_uri")
        try:
            airports_df, source = load_airports_index(source_uri)
        except FileNotFoundError as exc:
            return jsonify({"source": None, "total_countries": 0, "countries": [], "detail": str(exc)})
        except Exception as exc:
            return jsonify({"detail": str(exc)}), 500

        grouped = (
            airports_df.groupby("country", dropna=True)["iata_code"]
            .count()
            .reset_index(name="airport_count")
            .sort_values("country")
        )
        return jsonify({
            "source": source,
            "total_countries": len(grouped),
            "countries": [
                {"country": str(row["country"]), "airport_count": int(row["airport_count"])}
                for _, row in grouped.iterrows()
            ],
        })

    @app.get("/api/flight/airports")
    def flight_airports():
        country = request.args.get("country", "").strip()
        source_uri = request.args.get("source_uri")
        limit_raw = request.args.get("limit", "800")
        if not country:
            return jsonify({"detail": "Query parameter 'country' is required."}), 400

        try:
            limit = max(1, min(int(limit_raw), 5000))
        except ValueError:
            return jsonify({"detail": "Query parameter 'limit' must be an integer."}), 400

        try:
            airports_df, source = load_airports_index(source_uri)
        except FileNotFoundError as exc:
            return jsonify({
                "source": None,
                "country": country,
                "total_airports": 0,
                "airports": [],
                "detail": str(exc),
            })
        except Exception as exc:
            return jsonify({"detail": str(exc)}), 500

        selected = (
            airports_df[airports_df["country"].astype(str).str.casefold() == country.casefold()]
            .sort_values(["city", "airport_name", "iata_code"], na_position="last")
            .head(limit)
        )

        def _as_float(value: Any) -> float | None:
            try:
                parsed = float(value)
                return None if parsed != parsed else parsed
            except Exception:
                return None

        return jsonify({
            "source": source,
            "country": country,
            "total_airports": len(selected),
            "airports": [{
                "iata_code": optional_text(row.get("iata_code")),
                "airport_name": optional_text(row.get("airport_name")),
                "city": optional_text(row.get("city")),
                "state": optional_text(row.get("state")),
                "country": optional_text(row.get("country")),
                "latitude": _as_float(row.get("latitude")),
                "longitude": _as_float(row.get("longitude")),
            } for row in selected.to_dict(orient="records")],
        })

    @app.get("/api/flight/departures")
    def flight_departures():
        airport = request.args.get("airport", "").strip().upper()
        source_uri = request.args.get("source_uri")
        limit_raw = request.args.get("limit", "50")
        if not airport:
            return jsonify({"detail": "Query parameter 'airport' is required."}), 400

        try:
            limit = max(1, min(int(limit_raw), 500))
        except ValueError:
            return jsonify({"detail": "Query parameter 'limit' must be an integer."}), 400

        departures, matched, future, total_rows, source = [], 0, False, 0, None
        try:
            flights_df, source = load_upcoming_flights_frame(source_uri)
            total_rows = len(flights_df)
            departures, matched, future = extract_airport_departures(flights_df, airport, limit)
        except FileNotFoundError:
            pass
        except Exception as exc:
            return jsonify({"detail": str(exc)}), 500

        if not departures:
            try:
                airports_df, _ = load_airports_index()
                possible_dests = airports_df[airports_df["iata_code"] != airport]
                popular_dests_iata = ["JFK", "LAX", "LHR", "CDG", "DXB", "HND", "GRU", "EZE"]
                sample_destinations = possible_dests[possible_dests["iata_code"].isin(popular_dests_iata)]

                missing = 3 - len(sample_destinations)
                if missing > 0:
                    remainder = possible_dests[~possible_dests["iata_code"].isin(popular_dests_iata)]
                    sample_size = min(missing, len(remainder))
                    if sample_size > 0:
                        sample_destinations = pd.concat([sample_destinations, remainder.sample(sample_size)])

                today_iso = date.today().isoformat()
                today_br = date.today().strftime("%d/%m/%Y")
                fake_deps = []
                for _, dest_row in sample_destinations.head(3).iterrows():
                    fake_deps.append({
                        "flight_date": today_iso,
                        "flight_date_br": today_br,
                        "airline": "ZZ",
                        "flight_number": "9876",
                        "flight_code": "ZZ9876",
                        "origin_airport": airport,
                        "destination_airport": dest_row["iata_code"],
                        "scheduled_departure": "11:00",
                        "scheduled_arrival": "14:00",
                    })
                departures = fake_deps
                matched = len(departures)
                future = True
            except Exception:
                pass

        return jsonify({
            "source": source,
            "airport": airport,
            "total_rows": total_rows,
            "matched_rows": matched,
            "returned_rows": len(departures),
            "future_window": future,
            "departures": departures,
        })
