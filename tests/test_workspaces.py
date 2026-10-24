"""Workspace tab bookkeeping (modules/workspaces.py): plain dicts, no GPU or Gradio needed."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules import workspaces as ws  # noqa: E402


def test_add_switch_and_save():
    state = ws.new_state()
    state = ws.save_active(state, ["a", 1], ["img"])
    assert ws.active_tab(state)["v"] == ["a", 1] and ws.active_tab(state)["m"] == ["img"]
    state = ws.add_tab(state)
    assert state["active"] == "2" and ws.active_tab(state)["v"] is None  # a new tab starts at the defaults
    state = ws.switch(ws.save_active(state, ["b", 2], []), "1")
    assert ws.active_tab(state)["v"] == ["a", 1] and ws.find(state, "2")["v"] == ["b", 2]
    assert ws.choices(state) == [("Workspace 1", "1"), ("Workspace 2", "2")]
    assert ws.switch(state, "nope") is state


def test_save_does_not_mutate_the_previous_state():
    before = ws.new_state()
    ws.save_active(before, ["x"], [])
    assert before["tabs"][0]["v"] is None


def test_close_picks_the_left_neighbour_and_never_empties():
    state = ws.add_tab(ws.add_tab(ws.new_state()))  # 1, 2, 3 (3 active)
    state = ws.close_tab(ws.switch(state, "2"), "2")
    assert [t["id"] for t in state["tabs"]] == ["1", "3"] and state["active"] == "1"
    state = ws.close_tab(ws.close_tab(state, "1"), "3")
    assert len(state["tabs"]) == 1 and state["tabs"][0]["id"] == "4" and state["active"] == "4"
    assert ws.close_tab(state, "zzz") == state


def test_limit_and_rename():
    state = ws.new_state()
    for _ in range(ws.MAX_WORKSPACES - 1):
        state = ws.add_tab(state)
    assert len(state["tabs"]) == ws.MAX_WORKSPACES and ws.add_tab(state) is None
    state = ws.rename_tab(state, "1", "  Fox   walk  ")
    assert ws.find(state, "1")["name"] == "Fox walk"
    assert ws.rename_tab(state, "1", "   ") is state and ws.rename_tab(state, "9999", "x") is state
    assert len(ws.rename_tab(state, "1", "n" * 100)["tabs"][0]["name"]) == 40


def test_browser_round_trip_drops_media_and_stale_layouts():
    state = ws.add_tab(ws.save_active(ws.new_state(), ["a", 1], ["img"]))
    saved = ws.to_browser(state, ["b", 2])
    assert all("m" not in t for t in saved["tabs"]) and saved["tabs"][1]["v"] == ["b", 2]
    restored = ws.from_browser(saved, 2)
    assert restored["active"] == "2" and restored["counter"] == 2 and restored["tabs"][0]["v"] == ["a", 1]
    assert restored["tabs"][0]["m"] is None
    # Saved with a different number of controls: back to the defaults
    assert ws.from_browser(saved, 3)["tabs"][0]["v"] is None
    for junk in (None, {}, {"tabs": []}, {"tabs": [{"id": 1}]}, "x"):
        assert ws.from_browser(junk, 2) is None
