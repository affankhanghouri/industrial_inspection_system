from datetime import timedelta
import asyncio

import httpx

from industrial_safety.api import create_app
from industrial_safety.events.service import EventService
from industrial_safety.events.types import EventStatus
from industrial_safety.rules import Rule, RuleEngine, RuleType
from industrial_safety.rules.types import RuleMatch
from industrial_safety.storage import SQLiteEventStore
from industrial_safety.zones import (
    Zone,
    ZoneEngine,
    ZoneTransition,
    ZoneTransitionType,
)

from inference_helpers import frame
from ppe_helpers import NOW, person_track, tracking_result


def zone_rule() -> Rule:
    return Rule(
        "RULE-X", "1.0", "CAM-2", RuleType.LOADING_AREA_ENTRY,
        "LOADING_AREA_ENTRY", "high", True, 60.0, zone_id="loading",
    )


def rule_match(*, detected_at=NOW) -> RuleMatch:
    rule = zone_rule()
    zone = Zone(
        "loading", "CAM-2", "Loading", "loading_area",
        ((0, 0), (15, 0), (15, 15), (0, 15)), True, 1, 1,
    )
    track = person_track(2, (2, 2, 8, 12), camera_id="CAM-2")
    transition = ZoneTransition(
        "CAM-2", 10, zone, track, ZoneTransitionType.ENTER, detected_at
    )
    return RuleMatch(
        rule, "CAM-2", track, detected_at, "loading", .9, 1,
        zone_transition=transition,
    )


def test_event_creation_deduplication_and_sqlite_restart(tmp_path) -> None:
    database = tmp_path / "events.db"
    evidence = tmp_path / "evidence"
    store = SQLiteEventStore(database)
    service = EventService(
        store, evidence, person_model_version="person-v1", ppe_model_version="ppe-v1"
    )
    created = service.create(rule_match(), frame("CAM-2", 10))
    assert created is not None
    assert (evidence / created.snapshot_path).is_file()
    assert service.create(
        rule_match(detected_at=NOW + timedelta(seconds=30)), frame("CAM-2", 11)
    ) is None
    store.close()

    reopened = SQLiteEventStore(database)
    assert reopened.get(created.event_id) == created
    reopened.close()


def test_api_health_events_resolve_and_dismiss(tmp_path) -> None:
    store = SQLiteEventStore(tmp_path / "events.db")
    service = EventService(
        store, tmp_path / "evidence", person_model_version="p", ppe_model_version="h"
    )
    first = service.create(rule_match(), frame("CAM-2", 10))
    second_match = rule_match(detected_at=NOW + timedelta(seconds=61))
    second = service.create(second_match, frame("CAM-2", 11))
    assert first and second
    app = create_app(
        store,
        tmp_path / "evidence",
        camera_health_provider=lambda: {"CAM-2": {"camera_id": "CAM-2"}},
        web_root=__import__("pathlib").Path(__file__).resolve().parents[1] / "web",
    )
    async def exercise_api() -> None:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            assert (await client.get("/api/health")).json()["status"] == "ok"
            assert len((await client.get("/api/events")).json()) == 2
            resolved = await client.post(
                f"/api/events/{first.event_id}/resolve",
                json={"reason": "corrected"},
            )
            assert resolved.json()["status"] == EventStatus.RESOLVED.value
            dismissed = await client.post(
                f"/api/events/{second.event_id}/dismiss",
                json={"reason": "false positive"},
            )
            assert dismissed.json()["status"] == EventStatus.DISMISSED.value

    asyncio.run(exercise_api())
    assert store.stats()["open"] == 0
    store.close()


def test_full_mocked_zone_to_rule_to_persisted_event(tmp_path) -> None:
    zone = Zone(
        "loading", "CAM-2", "Loading", "loading_area",
        ((10, 10), (30, 10), (30, 30), (10, 30)), True, 1, 1,
    )
    zones = ZoneEngine((zone,))
    rules = RuleEngine((zone_rule(),))
    outside = person_track(7, (3, 3, 7, 9), camera_id="CAM-2", frame_id=1)
    inside = person_track(7, (18, 10, 22, 20), camera_id="CAM-2", frame_id=2)
    zones.evaluate(tracking_result((outside,), camera_id="CAM-2", frame_id=1))
    evaluation = zones.evaluate(
        tracking_result((inside,), camera_id="CAM-2", frame_id=2)
    )
    matches = rules.evaluate_zone(evaluation.transitions[0])
    store = SQLiteEventStore(tmp_path / "events.db")
    service = EventService(
        store, tmp_path / "evidence", person_model_version="p", ppe_model_version="h"
    )
    event = service.create(matches[0], frame("CAM-2", 2))
    assert event is not None
    assert store.get(event.event_id) is not None
    store.close()


def test_event_api_includes_and_serves_mp4_evidence(tmp_path) -> None:
    class Recorder:
        def trigger(self, *_args, **_kwargs) -> bool:
            return True

    evidence = tmp_path / "evidence"
    store = SQLiteEventStore(tmp_path / "events.db")
    service = EventService(
        store,
        evidence,
        person_model_version="p",
        ppe_model_version="h",
        evidence_recorder=Recorder(),  # type: ignore[arg-type]
    )
    event = service.create(rule_match(), frame("CAM-2", 10))
    assert event is not None and event.clip_path is not None
    clip = evidence / event.clip_path
    clip.parent.mkdir(parents=True, exist_ok=True)
    clip.write_bytes(b"small-mp4-test")
    app = create_app(store, evidence)

    async def exercise_api() -> None:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            payload = (await client.get(f"/api/events/{event.event_id}")).json()
            assert payload["clip_path"] == event.clip_path
            assert payload["clip_url"] == f"/api/events/{event.event_id}/clip"
            response = await client.get(payload["clip_url"])
            assert response.status_code == 200
            assert response.headers["content-type"] == "video/mp4"
            assert response.content == b"small-mp4-test"

    asyncio.run(exercise_api())
    store.close()


def test_dashboard_gracefully_describes_event_without_clip() -> None:
    script = (__import__("pathlib").Path(__file__).resolve().parents[1] / "web/static/app.js").read_text()
    assert "No video clip is available for this event." in script


def test_dashboard_event_close_control_dismisses_and_hides_case() -> None:
    script = (__import__("pathlib").Path(__file__).resolve().parents[1] / "web/static/app.js").read_text()
    assert 'class="event-dismiss"' in script
    assert "`/api/events/${id}/dismiss`" in script
    assert "selectedStatus==='ALL'&&event.status!=='DISMISSED'" in script
