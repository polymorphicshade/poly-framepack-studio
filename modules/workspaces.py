"""Workspace tabs for the Generate page: each tab remembers its own settings (generation type,
prompts, sliders, LoRAs, ...) and input images, and switching tabs swaps them into the one set of
controls. Plain dicts, so the Generate page keeps them in a gr.State and, minus the images, in the
browser (gr.BrowserState) to survive a reload.

A tab is {"id": str, "name": str, "v": list | None, "m": list | None}: "v" the JSON-able control
values and "m" the media values (images, video), both in the order the page lists its controls;
None means the controls' defaults.
"""

import copy

MAX_WORKSPACES = 12
STORAGE_KEY = "poly-framepack-studio.workspaces"


def new_state():
    return {"tabs": [{"id": "1", "name": "Workspace 1", "v": None, "m": None}], "active": "1", "counter": 1}


def find(state, tab_id):
    return next((t for t in state["tabs"] if t["id"] == tab_id), None)


def active_tab(state):
    return find(state, state["active"]) or state["tabs"][0]


def choices(state):
    return [(t["name"], t["id"]) for t in state["tabs"]]


def save_active(state, values, media):
    """A copy of the state with the live control values stored in the active tab."""
    state = copy.copy(state)
    state["tabs"] = [dict(t) for t in state["tabs"]]
    tab = active_tab(state)
    tab["v"], tab["m"] = list(values), list(media)
    return state


def switch(state, tab_id):
    """Make ``tab_id`` the active tab (the live values must already be saved)."""
    if find(state, tab_id) is None:
        return state
    return {**state, "active": tab_id}


def add_tab(state):
    """Append an empty tab and make it active. Returns the state, or None at the limit."""
    if len(state["tabs"]) >= MAX_WORKSPACES:
        return None
    counter = state["counter"] + 1
    tab = {"id": str(counter), "name": f"Workspace {counter}", "v": None, "m": None}
    return {**state, "tabs": [*state["tabs"], tab], "active": tab["id"], "counter": counter}


def close_tab(state, tab_id):
    """Remove a tab; the one to its left (or a fresh one, if it was the last) becomes active."""
    index = next((i for i, t in enumerate(state["tabs"]) if t["id"] == tab_id), None)
    if index is None:
        return state
    tabs = [t for t in state["tabs"] if t["id"] != tab_id]
    if not tabs:
        return add_tab({**state, "tabs": []})
    active = state["active"]
    if active == tab_id or not any(t["id"] == active for t in tabs):
        active = tabs[max(0, index - 1)]["id"]
    return {**state, "tabs": tabs, "active": active}


def rename_tab(state, tab_id, name):
    name = " ".join(str(name or "").split())[:40]
    if not name or find(state, tab_id) is None:
        return state
    return {**state, "tabs": [{**t, "name": name} if t["id"] == tab_id else t for t in state["tabs"]]}


def to_browser(state, values):
    """What the browser keeps: names and settings, with the active tab's live values, no media."""
    state = save_active(state, values, [])
    return {"tabs": [{"id": t["id"], "name": t["name"], "v": t["v"]} for t in state["tabs"]],
            "active": state["active"], "counter": state["counter"]}


def from_browser(data, value_count):
    """The state for a page load from what the browser kept, or None if there is nothing usable.
    Tabs saved by a version with a different set of controls fall back to the defaults."""
    try:
        tabs = [{"id": str(t["id"]), "name": str(t["name"]),
                 "v": t["v"] if isinstance(t.get("v"), list) and len(t["v"]) == value_count else None, "m": None}
                for t in data["tabs"]][:MAX_WORKSPACES]
        if not tabs:
            return None
        ids = [t["id"] for t in tabs]
        active = str(data.get("active"))
        counter = max(int(data.get("counter", 0)), *(int(i) for i in ids if i.isdigit()), 0)
        return {"tabs": tabs, "active": active if active in ids else ids[0], "counter": counter}
    except (KeyError, TypeError, ValueError, AttributeError):
        return None
