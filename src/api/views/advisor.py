from __future__ import annotations

from typing import Any

from flask import Flask, jsonify, request


MAX_SESSION_MESSAGES = 40

# ---------------------------------------------------------------------------
# Fields that can be requested from the user when missing from the payload.
# Maps payload attribute name → human-friendly follow-up question.
# Only fields that exist in the S3 dataset should be listed here.
# ---------------------------------------------------------------------------
S3_AVAILABLE_FIELDS: list[str] = [
    "origin_airport",
    "destination_airport",
]

_FIELD_QUESTIONS: dict[str, str] = {
    "origin_airport":      "Which airport are you departing from? (e.g. GRU, JFK)",
    "destination_airport": "What is your destination airport? (e.g. MIA, LHR)",
    "cabin_class":         "Which cabin class — economy, business, or first?",
}

_FIELD_QUESTIONS = {key: value for key, value in _FIELD_QUESTIONS.items() if key in S3_AVAILABLE_FIELDS}

# ---------------------------------------------------------------------------
# System prompt sent to the LLM on every call.
# Defines language, tone, and the probing strategy.
# ---------------------------------------------------------------------------
ADVISOR_SYSTEM_PROMPT = """
You are a friendly and practical flight advisor.

## Language
- Reply in the same language used by the user.
- If the language is unclear, default to Brazilian Portuguese.

## Discovery-first behaviour
- If the request is open, exploratory, or about destination planning, do not jump into delay metrics.
- First understand what the traveler actually needs.
- In those cases, answer in a warmer and more consultative tone.
- If destination or region context exists, you may comment on the typical climate, regional profile,
  best season in general terms, and tourist activities or travel styles that fit the place.
- Treat climate or weather as general regional guidance, not live weather, unless real-time data is provided.

## Delay assessment
- Only provide delay probability, risk level, top factors, or lower-risk alternatives when the user
  explicitly asks about delay, punctuality, or operational risk and the context is sufficient.
- When the question is not explicitly about delay, avoid numbers or operational delay metrics.

## Probing strategy
- When the user's request is missing key information for a specific analysis, ask for the missing details
  naturally, one or two questions at a time.

Key fields required for a route-level delay assessment:
- Departure airport (IATA code)
- Destination airport (IATA code)

Optional fields that help narrow the result:
- Airline
- Scheduled departure time
- Travel date

If the context includes a "probing_questions" list, use those questions as a guide for what to ask next.
Never dump all questions at once and prioritise the most relevant missing field.

## Flight search results
- If real-time flight search results are provided in the context, incorporate them naturally.
- Highlight price, availability, and schedule only when those data are actually present.

## Complete travel guide requests
- If the user asks for a complete travel guide, a full destination guide, or explicitly asks for climate,
  attractions, gastronomy, flights, and accommodation in the same answer, provide a complete long-form response.
- Never leave a section unfinished and never stop mid-structure.
- If you start a list of topics, cover all requested topics fully.
- End with a conclusion or a clear next-step question.
- If the user explicitly asks for English, respond in English.
- Use this structure for complete travel guides:
  1. General destination info, including climate and best time to visit
  2. Top tourist attractions, with at least 5 concrete suggestions
  3. Practical details, including flight tips, accommodation guidance, and local transportation
  4. Cultural and food tips
  5. Conclusion with next steps
- If live flight prices, hotel inventory, or current availability are not present in the structured context,
  say that clearly and provide general planning guidance instead of inventing data.

## Tone
Friendly, helpful, and well-structured. Avoid jargon.
"""


def build_probing_questions(
    payload: Any,
    available_fields: list[str] | None = None,
) -> list[str]:
    """
    Returns a prioritised list of follow-up questions for fields that are
    missing from *payload* but are available in the S3 dataset.

    Only the first two questions should ever be shown to the user at once —
    the caller is responsible for slicing if needed.

    Args:
        payload:          The validated AdviseRequest instance.
        available_fields: S3 field names to consider. Defaults to S3_AVAILABLE_FIELDS.

    Returns:
        Ordered list of question strings (most critical first).
    """
    fields = available_fields if available_fields is not None else S3_AVAILABLE_FIELDS
    questions: list[str] = []
    for field in fields:
        value = getattr(payload, field, None)
        if not value and field in _FIELD_QUESTIONS:
            questions.append(_FIELD_QUESTIONS[field])
    return questions


def register_advisor_views(app: Flask, deps: dict[str, Any]) -> None:
    if "advisor_history" in app.view_functions:
        return

    # -----------------------------------------------------------------------
    # Unpack dependencies
    # -----------------------------------------------------------------------
    AdviseRequest                  = deps["AdviseRequest"]
    AdviseResponse                 = deps["AdviseResponse"]
    ValidationError                = deps["ValidationError"]
    coerce_feature_types           = deps["coerce_feature_types"]
    load_advisor_messages          = deps["load_advisor_messages"]
    persist_advisor_messages       = deps["persist_advisor_messages"]
    reset_advisor_messages         = deps["reset_advisor_messages"]
    trim_chat_text                 = deps["trim_chat_text"]
    should_run_route_prediction    = deps["should_run_route_prediction"]
    should_run_weekly_route_prediction = deps["should_run_weekly_route_prediction"]
    enrich_route_payload_from_question = deps["enrich_route_payload_from_question"]
    route_updates_from_request     = deps["route_updates_from_request"]
    user_chat_message              = deps["user_chat_message"]
    build_discovery_response_runtime = deps["build_discovery_response_runtime"]
    should_use_llm                 = deps.get("should_use_llm") or deps["should_use_nemotron"]
    generate_llm_advice            = deps.get("generate_llm_advice") or deps["generate_nemotron_advice"]
    build_discovery_context        = deps["build_discovery_context"]
    assistant_chat_message         = deps["assistant_chat_message"]
    load_assets                    = deps["load_assets"]
    compute_model_delay_snapshot   = deps["compute_model_delay_snapshot"]
    build_weekly_route_prediction  = deps["build_weekly_route_prediction"]
    advice_text                    = deps["advice_text"]
    build_advice_context           = deps["build_advice_context"]

    # Optional dependencies
    search_flights              = deps.get("search_flights")
    serialize_messages_for_llm  = deps.get("serialize_messages_for_llm")

    # S3 fields available for probing (can be overridden via deps)
    available_s3_fields: list[str] = deps.get("available_s3_fields", S3_AVAILABLE_FIELDS)

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _serialize_history(messages: list) -> list[dict]:
        """
        Converts internal Pydantic history objects to the
        {"role": ..., "content": ...} format expected by the LLM.
        Uses deps-provided serializer when available.
        """
        if serialize_messages_for_llm:
            return serialize_messages_for_llm(messages)
        result = []
        for msg in messages:
            if hasattr(msg, "model_dump"):
                d = msg.model_dump()
            elif hasattr(msg, "__dict__"):
                d = msg.__dict__
            else:
                d = dict(msg)
            role    = d.get("role")
            content = d.get("content") or d.get("text") or ""
            if role and content:
                result.append({"role": role, "content": str(content)})
        return result

    def _trim_session_messages(
        messages: list,
        max_messages: int = MAX_SESSION_MESSAGES,
    ) -> list:
        """
        Keeps only the most recent *max_messages* entries so the session
        does not grow unboundedly.
        """
        if len(messages) > max_messages:
            app.logger.debug(
                "Session history trimmed from %d to %d messages.",
                len(messages),
                max_messages,
            )
            return messages[-max_messages:]
        return messages

    def _run_flight_search(payload: Any) -> tuple[list[dict], str | None]:
        """
        Executes the optional flight search and returns (results, error).
        Returns ([], None) when search_flights is not configured.
        """
        if search_flights is None:
            return [], None

        origin        = getattr(payload, "origin_airport", None) or getattr(payload, "origin", None)
        destination   = getattr(payload, "destination_airport", None) or getattr(payload, "destination", None)
        departure_date = getattr(payload, "departure_date", None)
        passengers    = getattr(payload, "passengers", 1) or 1
        cabin_class   = getattr(payload, "cabin_class", "economy") or "economy"

        if not (origin or destination):
            return [], None

        try:
            results = search_flights(
                origin=origin,
                destination=destination,
                departure_date=departure_date,
                passengers=passengers,
                cabin_class=cabin_class,
            ) or []
            app.logger.info(
                "search_flights returned %d result(s) for %s → %s.",
                len(results),
                origin,
                destination,
            )
            return results, None
        except Exception as exc:
            app.logger.warning("Flight search failed: %s", exc)
            return [], str(exc)

    def _build_flight_search_ctx(
        results: list[dict],
        error: str | None,
    ) -> dict:
        """Packages flight search outcome for injection into LLM context."""
        return {
            "enabled": search_flights is not None,
            "results": results,
            "error":   error,
        }

    # -----------------------------------------------------------------------
    # Endpoints
    # -----------------------------------------------------------------------

    @app.get("/api/advisor/history")
    def advisor_history():
        session_id, messages = load_advisor_messages()
        messages = persist_advisor_messages(session_id, messages)
        return jsonify({
            "session_id": session_id,
            "messages":   [item.model_dump() for item in messages],
        })

    @app.post("/api/advisor/reset")
    def advisor_reset():
        session_id, messages = reset_advisor_messages()
        return jsonify({
            "session_id": session_id,
            "messages":   [item.model_dump() for item in messages],
        })

    @app.route("/advise", methods=["POST", "OPTIONS"])
    def advise():
        if request.method == "OPTIONS":
            return ("", 204)

        # --- Parse & validate request body --------------------------------
        payload_data = request.get_json(silent=True)
        if payload_data is None:
            if request.get_data(as_text=True).strip():
                return jsonify({"detail": "Body must be valid JSON."}), 400
            payload_data = {}

        try:
            payload = AdviseRequest.model_validate(payload_data)
        except ValidationError as exc:
            return jsonify({"detail": exc.errors()}), 400
        payload = enrich_route_payload_from_question(payload)
        route_updates = route_updates_from_request(payload)

        # --- Session & user message ----------------------------------------
        session_id, messages = load_advisor_messages()

        user_text = trim_chat_text(payload.question)
        if not user_text and should_run_route_prediction(payload):
            # Synthetic message so the LLM has something to respond to
            user_text = (
                f"Analyze the route {payload.origin_airport} to "
                f"{payload.destination_airport} with airline {payload.airline}, "
                f"scheduled departure {int(payload.scheduled_departure):04d}."
            )
        if user_text:
            messages.append(user_chat_message(user_text))

        # --- Optional flight search ----------------------------------------
        flight_results, flight_search_error = _run_flight_search(payload)
        flight_search_ctx = _build_flight_search_ctx(flight_results, flight_search_error)

        # --- Probing questions (missing S3-available fields) ---------------
        probing_questions = build_probing_questions(payload, available_s3_fields)

        # Serialised history for the LLM
        llm_history = _serialize_history(messages)

        if should_run_weekly_route_prediction(payload):
            try:
                pipeline, meta, maps, global_rate, route_distance_map, global_distance = load_assets()
            except Exception as exc:
                return jsonify({"detail": str(exc)}), 500

            weekly_response = build_weekly_route_prediction(payload, pipeline, meta, limit=3)
            if weekly_response is None:
                try:
                    snapshot = compute_model_delay_snapshot(
                        payload,
                        pipeline,
                        meta,
                        maps,
                        global_rate,
                        route_distance_map,
                        global_distance,
                        limit=3,
                    )
                except ValueError as exc:
                    return jsonify({"detail": str(exc)}), 400

                fallback_advice = advice_text(
                    snapshot["delay_probability"],
                    snapshot["risk_level"],
                    snapshot["top_factors"],
                    snapshot["suggested_flights"],
                    snapshot["delay_prediction"],
                )
                advice = fallback_advice
                advice_source = "model_fallback"
                advice_model = None

                if should_use_llm(user_text):
                    advice_ctx = build_advice_context(
                        payload,
                        snapshot["delay_probability"],
                        snapshot["risk_level"],
                        snapshot["delay_prediction"],
                        snapshot["top_factors"],
                        fallback_advice,
                        snapshot["suggested_flights"],
                        messages,
                    )
                    advice_ctx["flight_search"] = flight_search_ctx

                    try:
                        llm_result = generate_llm_advice(
                            advice_ctx,
                            history=llm_history,
                            system_prompt=ADVISOR_SYSTEM_PROMPT,
                        )
                        advice = llm_result.content
                        advice_source = llm_result.provider
                        advice_model = llm_result.model
                    except RuntimeError as exc:
                        app.logger.warning("LLM unavailable, falling back to sparse model advice: %s", exc)
                        advice_source = "model_fallback"

                weekly_response = AdviseResponse(
                    delay_probability=snapshot["delay_probability"],
                    delay_prediction=snapshot["delay_prediction"],
                    risk_level=snapshot["risk_level"],
                    top_factors=snapshot["top_factors"],
                    advice=advice,
                    advice_source=advice_source,
                    advice_model=advice_model,
                    suggested_flights=snapshot["suggested_flights"],
                    mode="route_model",
                )

            if weekly_response is not None:
                weekly_response = weekly_response.model_copy(update={
                    "route_updates": route_updates,
                })
                messages.append(assistant_chat_message(weekly_response))
                messages = _trim_session_messages(messages)
                messages = persist_advisor_messages(session_id, messages)
                weekly_response = weekly_response.model_copy(update={
                    "session_id": session_id,
                    "messages": messages,
                    "route_updates": route_updates,
                })
                return jsonify(weekly_response.model_dump())

        # ==================================================================
        # DISCOVERY PATH — incomplete route context
        # ==================================================================
        if not should_run_route_prediction(payload):
            discovery = build_discovery_response_runtime(payload).model_copy(update={
                "route_updates": route_updates,
            })

            if should_use_llm(user_text):
                discovery_ctx = build_discovery_context(payload, discovery.advice, messages)
                discovery_ctx["flight_search"]      = flight_search_ctx
                # Provide the LLM with what to ask next
                discovery_ctx["probing_questions"]  = probing_questions[:2]  # at most 2 at once

                try:
                    llm_result = generate_llm_advice(
                        discovery_ctx,
                        history=llm_history,
                        system_prompt=ADVISOR_SYSTEM_PROMPT,
                    )
                    discovery = discovery.model_copy(update={
                        "advice":        llm_result.content,
                        "advice_source": llm_result.provider,
                        "advice_model":  llm_result.model,
                    })
                except RuntimeError as exc:
                    app.logger.warning("LLM unavailable, falling back to discovery heuristic: %s", exc)
                    discovery = discovery.model_copy(update={
                        "advice_source": "heuristic_fallback",
                        "advice_model":  None,
                    })
            else:
                # Heuristic-only discovery path — annotate source clearly
                discovery = discovery.model_copy(update={
                    "advice_source": "heuristic",
                    "advice_model":  None,
                })

            messages.append(assistant_chat_message(discovery))
            messages = _trim_session_messages(messages)
            messages = persist_advisor_messages(session_id, messages)
            discovery = discovery.model_copy(update={
                "session_id":        session_id,
                "messages":          messages,
                "probing_questions": probing_questions,  # expose to frontend
                "route_updates":     route_updates,
            })
            return jsonify(discovery.model_dump())

        # ==================================================================
        # FULL PIPELINE PATH — complete route context available
        # ==================================================================
        try:
            pipeline, meta, maps, global_rate, route_distance_map, global_distance = load_assets()
        except Exception as exc:
            return jsonify({"detail": str(exc)}), 500

        try:
            snapshot = compute_model_delay_snapshot(
                payload,
                pipeline,
                meta,
                maps,
                global_rate,
                route_distance_map,
                global_distance,
                limit=3,
            )
        except ValueError as exc:
            return jsonify({"detail": str(exc)}), 400
        prob              = snapshot["delay_probability"]
        delay_prediction  = snapshot["delay_prediction"]
        level             = snapshot["risk_level"]
        top               = snapshot["top_factors"]
        suggested_flights = snapshot["suggested_flights"]

        # Heuristic advice now also receives flight_results so the fallback
        # path can surface real-time alternatives even without the LLM.
        fallback_advice = advice_text(prob, level, top, suggested_flights, delay_prediction)

        advice        = fallback_advice
        advice_source = "heuristic"
        advice_model  = None

        if should_use_llm(user_text):
            advice_ctx = build_advice_context(
                payload, prob, level, delay_prediction, top, fallback_advice, suggested_flights, messages
            )
            advice_ctx["flight_search"]   = flight_search_ctx

            try:
                llm_result = generate_llm_advice(
                    advice_ctx,
                    history=llm_history,
                    system_prompt=ADVISOR_SYSTEM_PROMPT,
                )
                advice        = llm_result.content
                advice_source = llm_result.provider   # e.g. "nemotron", "openai", ...
                advice_model  = llm_result.model
            except RuntimeError as exc:
                app.logger.warning("LLM unavailable, falling back to heuristic advice: %s", exc)
                advice_source = "heuristic_fallback"

        response_model = AdviseResponse(
            delay_probability = prob,
            delay_prediction  = delay_prediction,
            risk_level        = level,
            top_factors       = top,
            advice            = advice,
            advice_source     = advice_source,
            advice_model      = advice_model,
            suggested_flights = suggested_flights,
            route_updates     = route_updates,
        )
        messages.append(assistant_chat_message(response_model))
        messages = _trim_session_messages(messages)
        messages = persist_advisor_messages(session_id, messages)
        response_model = response_model.model_copy(update={
            "session_id": session_id,
            "messages":   messages,
            "route_updates": route_updates,
        })
        return jsonify(response_model.model_dump())
