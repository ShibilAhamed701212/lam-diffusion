"""CPU tests for the planner, spec and API layer (no model weights needed)."""
import pytest

from planner import rule_plan
from prompting import build_prompt
from spec import normalize_spec


def test_rule_plan_matches_whole_words_only():
    spec = rule_plan("a knight with a sword at a castle")
    assert spec["scene"] == "castle"  # "knight" must not add the "night" scene
    assert spec["props"] == ["sword"]
    assert rule_plan("a girl visiting a cafe")["characters"][0]["pose"] == "standing"  # not "sit"
    assert rule_plan("a boy who likes to catch fish")["props"] == []  # not "cat"


def test_rule_plan_keeps_inflected_forms():
    spec = rule_plan("two girls sitting with their cats under cherry blossoms")
    assert [c["pose"] for c in spec["characters"]] == ["sitting", "sitting"]
    assert spec["props"] == ["cat"]
    assert spec["scene"] == "cherry blossom"


def test_rule_plan_attributes_and_prompt():
    spec = rule_plan("an old man with white hair waving")
    ch = spec["characters"][0]
    assert (ch["gender"], ch["age"], ch["pose"], ch["hair_color"]) == ("male", "elderly", "waving", "white")
    prompt, negative = build_prompt(spec)
    assert prompt.startswith("1boy, solo") and "white hair" in prompt
    assert "1girl" in negative


def test_normalize_spec_clamps_and_rejects_bad_boxes():
    spec = normalize_spec({"characters": [{"bbox": [0.9, 0.1, 0.1, 0.9], "gender": "girl"}] * 5})
    assert len(spec["characters"]) == 3
    ch = spec["characters"][0]
    assert ch["gender"] == "female"
    x0, y0, x1, y1 = ch["bbox"]  # the inverted box is replaced by a layout box
    assert 0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1


def test_descriptor_to_spec_maps_actions_to_pose():
    lam_engine = pytest.importorskip("lam_engine")
    spec = lam_engine.descriptor_to_spec({
        "scene": "park", "aspect": "portrait",
        "characters": [{"gender": "male", "age": "elderly", "actions": ["standing", "waving", "smiling"]}],
    })
    ch = spec["characters"][0]
    assert (ch["gender"], ch["age"], ch["pose"]) == ("male", "elderly", "standing")
    assert ch["action"] == "waving, smiling"
    assert lam_engine.descriptor_to_spec(None) is None
    assert lam_engine.descriptor_to_spec({"characters": "x"}) is None


@pytest.fixture
def client(monkeypatch):
    testclient = pytest.importorskip("fastapi.testclient")
    import app

    monkeypatch.setattr(app, "TOKEN", "secret")
    return testclient.TestClient(app.app)


def test_api_requires_token(client):
    assert client.post("/plan", json={"prompt": "a girl"}).status_code == 401
    assert client.post("/plan", json={"prompt": "a girl"}, headers={"x-lam-token": "nope"}).status_code == 401


def test_api_plan_with_rules_and_safety_gate(client):
    h = {"x-lam-token": "secret"}
    r = client.post("/plan", json={"prompt": "two girls on a beach", "planner": "rules"}, headers=h)
    assert r.status_code == 200
    body = r.json()
    assert body["planner_used"] == "rules" and len(body["spec"]["characters"]) == 2 and body["pose_image"]
    assert client.post("/plan", json={"prompt": "a nude girl", "planner": "rules"}, headers=h).status_code == 400
    assert client.post("/generate", json={"spec": {}}, headers=h).status_code == 503  # models not loaded
