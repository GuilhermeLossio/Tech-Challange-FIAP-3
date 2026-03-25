from __future__ import annotations

from flask import Flask, render_template

_PAGES = [
    ("/",            "dashboard.html",   "Flight Advisor | Front",        "dashboard"),
    ("/front",       "dashboard.html",   "Flight Advisor | Front",        "dashboard"),
    ("/flight",      "flight.html",      "Flight Advisor | Flights",      "flight"),
    ("/flights",     "flight.html",      "Flight Advisor | Flights",      "flight"),
    ("/predictions", "predictions.html", "Flight Advisor | Predictions",  "predictions"),
    ("/advisor",     "advisor.html",     "Flight Advisor | Advisor",      "advisor"),
]


def register_page_views(app: Flask) -> None:
    if "home" in app.view_functions:
        return

    for path, template_name, title, active_page in _PAGES:
        endpoint = path.strip("/") or "home"

        def _make_view(tpl: str = template_name, page_title: str = title, active: str = active_page):
            def _view():
                return render_template(tpl, page_title=page_title, active_page=active)

            return _view

        app.add_url_rule(path, endpoint=endpoint, view_func=_make_view())
