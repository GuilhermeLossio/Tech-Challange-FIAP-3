from __future__ import annotations

from typing import Any

from flask import Flask, jsonify, request


def register_advisor_views(app: Flask, deps: dict[str, Any]) -> None:
    if "advisor_history" in app.view_functions:
        return

    AdviseRequest = deps["AdviseRequest"]
    AdviseResponse = deps["AdviseResponse"]
    ValidationError = deps["ValidationError"]
    coerce_feature_types = deps["coerce_feature_types"]
    load_advisor_messages = deps["load_advisor_messages"]
    persist_advisor_messages = deps["persist_advisor_messages"]
    reset_advisor_messages = deps["reset_advisor_messages"]
    trim_chat_text = deps["trim_chat_text"]
    has_full_route_context = deps["has_full_route_context"]
    user_chat_message = deps["user_chat_message"]
    build_discovery_response_runtime = deps["build_discovery_response_runtime"]
    should_use_nemotron = deps["should_use_nemotron"]
    generate_nemotron_advice = deps["generate_nemotron_advice"]
    build_discovery_context = deps["build_discovery_context"]
    assistant_chat_message = deps["assistant_chat_message"]
    load_assets = deps["load_assets"]
    build_features = deps["build_features"]
    risk_level = deps["risk_level"]
    compute_top_factors = deps["compute_top_factors"]
    build_suggested_flights = deps["build_suggested_flights"]
    advice_text = deps["advice_text"]
    build_advice_context = deps["build_advice_context"]

    @app.get("/api/advisor/history")
    def advisor_history():
        session_id, messages = load_advisor_messages()
        messages = persist_advisor_messages(session_id, messages)
        return jsonify({
            "session_id": session_id,
            "messages": [item.model_dump() for item in messages],
        })

    @app.post("/api/advisor/reset")
    def advisor_reset():
        session_id, messages = reset_advisor_messages()
        return jsonify({
            "session_id": session_id,
            "messages": [item.model_dump() for item in messages],
        })

    @app.route("/advise", methods=["POST", "OPTIONS"])
    def advise():
        if request.method == "OPTIONS":
            return ("", 204)

        payload_data = request.get_json(silent=True)
        if payload_data is None:
            if request.get_data(as_text=True).strip():
                return jsonify({"detail": "Body must be valid JSON."}), 400
            payload_data = {}

        try:
            payload = AdviseRequest.model_validate(payload_data)
        except ValidationError as exc:
            return jsonify({"detail": exc.errors()}), 400

        session_id, messages = load_advisor_messages()

        user_text = trim_chat_text(payload.question)
        if not user_text and has_full_route_context(payload):
            user_text = (
                f"Analise a rota {payload.origin_airport} para {payload.destination_airport} com a companhia "
                f"{payload.airline} e partida prevista para {int(payload.scheduled_departure):04d}."
            )
        if user_text:
            messages.append(user_chat_message(user_text))

        if not has_full_route_context(payload):
            discovery = build_discovery_response_runtime(payload)
            if should_use_nemotron(user_text):
                try:
                    llm_result = generate_nemotron_advice(
                        build_discovery_context(payload, discovery.advice, messages)
                    )
                    discovery = discovery.model_copy(update={
                        "advice": llm_result.content,
                        "advice_source": llm_result.provider,
                        "advice_model": llm_result.model,
                    })
                except RuntimeError as exc:
                    app.logger.warning("Falling back to discovery advisor text: %s", exc)

            messages.append(assistant_chat_message(discovery))
            messages = persist_advisor_messages(session_id, messages)
            discovery = discovery.model_copy(update={
                "session_id": session_id,
                "messages": messages,
            })
            return jsonify(discovery.model_dump())

        try:
            pipeline, meta, maps, global_rate, route_distance_map, global_distance = load_assets()
        except Exception as exc:
            return jsonify({"detail": str(exc)}), 500

        try:
            df = build_features(payload, maps, global_rate, route_distance_map, global_distance)
        except ValueError as exc:
            return jsonify({"detail": str(exc)}), 400

        X = df
        if meta and "features" in meta and "selected" in meta["features"]:
            required = meta["features"]["selected"]
            missing = [column for column in required if column not in df.columns]
            if missing:
                return jsonify({"detail": f"Missing columns required by the model: {', '.join(missing)}"}), 400
            X = coerce_feature_types(
                df[required],
                meta["features"].get("numeric", []),
                meta["features"].get("categorical", []),
            )

        prob = float(pipeline.predict_proba(X)[:, 1][0])
        level = risk_level(prob)
        top = compute_top_factors(df, global_rate)
        suggested_flights = build_suggested_flights(payload, pipeline, meta, limit=3)
        fallback_advice = advice_text(prob, level, top, suggested_flights)

        advice = fallback_advice
        advice_source = "heuristic"
        advice_model = None
        if should_use_nemotron(user_text):
            try:
                llm_result = generate_nemotron_advice(build_advice_context(
                    payload, prob, level, top, fallback_advice, suggested_flights, messages
                ))
                advice = llm_result.content
                advice_source = llm_result.provider
                advice_model = llm_result.model
            except RuntimeError as exc:
                app.logger.warning("Falling back to heuristic advisor text: %s", exc)

        response_model = AdviseResponse(
            delay_probability=prob,
            risk_level=level,
            top_factors=top,
            advice=advice,
            advice_source=advice_source,
            advice_model=advice_model,
            suggested_flights=suggested_flights,
        )
        messages.append(assistant_chat_message(response_model))
        messages = persist_advisor_messages(session_id, messages)
        response_model = response_model.model_copy(update={
            "session_id": session_id,
            "messages": messages,
        })
        return jsonify(response_model.model_dump())
