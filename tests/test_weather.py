from datetime import date, timedelta

import pytest
import responses

from data.database import WeatherDaily, session_scope
from de.weather import loader as weather


def daily_payload(start: date, temps: list[float | None]) -> dict:
    days = [(start + timedelta(days=i)).isoformat() for i in range(len(temps))]
    return {"daily": {"time": days, "temperature_2m_mean": temps}}


@pytest.mark.parametrize(
    ("tavg", "expected"), [(65.0, (0.0, 0.0)), (40.0, (25.0, 0.0)), (80.5, (0.0, 15.5))]
)
def test_degree_days(tavg, expected):
    assert weather.degree_days(tavg, 65.0) == expected


@responses.activate
def test_ingest_weather_joins_archive_and_forecast(settings):
    today = date(2024, 1, 10)
    # Archive runs through Jan 3 (seven day lag), with the last day not yet processed
    responses.get(
        weather.ARCHIVE_URL,
        json=daily_payload(date(2024, 1, 1), [30.0, 31.0, None]),
    )
    # Forecast fills Jan 3 through today plus forecast_days (3)
    forecast_temps = [32.0 + i for i in range(11)]
    responses.get(weather.FORECAST_URL, json=daily_payload(date(2024, 1, 3), forecast_temps))

    count = weather.WeatherLoader(settings, today=today).run(date(2024, 1, 1), today)
    assert count == 13

    assert "end_date=2024-01-03" in responses.calls[0].request.url
    forecast_url = responses.calls[1].request.url
    assert "start_date=2024-01-03" in forecast_url
    assert "end_date=2024-01-13" in forecast_url

    with session_scope(settings.db_url) as session:
        jan_1 = session.get(WeatherDaily, (date(2024, 1, 1), "chicago"))
        jan_12 = session.get(WeatherDaily, (date(2024, 1, 12), "chicago"))
    assert jan_1.hdd == 35.0
    assert not jan_1.is_forecast
    assert jan_12.is_forecast
