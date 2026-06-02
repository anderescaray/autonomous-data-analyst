from src.agent import _make_harvester_node

PNG  = {"format": "png",  "data": b"\x89PNG"}
HTML = {"format": "html", "data": "<div>chart</div>"}

_EMPTY_STATE: dict = {
    "messages": [],
    "profile_text": "",
    "chart_plan": [],
    "chart_explanations": [],
    "captured_figures": [],
}


def test_harvests_single_figure():
    captured = [PNG]
    node = _make_harvester_node(captured)
    result = node(_EMPTY_STATE)
    assert result == {"captured_figures": [PNG]}


def test_drains_captured_after_harvest():
    captured = [PNG]
    node = _make_harvester_node(captured)
    node(_EMPTY_STATE)
    assert captured == []


def test_second_call_sees_only_new_figures():
    captured = [PNG]
    node = _make_harvester_node(captured)
    node(_EMPTY_STATE)
    captured.append(HTML)
    result = node(_EMPTY_STATE)
    assert result == {"captured_figures": [HTML]}


def test_harvests_multiple_figures():
    captured = [PNG, HTML]
    node = _make_harvester_node(captured)
    result = node(_EMPTY_STATE)
    assert result["captured_figures"] == [PNG, HTML]


def test_empty_captured_returns_no_update():
    captured = []
    node = _make_harvester_node(captured)
    result = node(_EMPTY_STATE)
    assert result == {}


def test_independent_captured_lists_do_not_interfere():
    captured_a = [PNG]
    captured_b = [HTML]
    node_a = _make_harvester_node(captured_a)
    node_b = _make_harvester_node(captured_b)
    assert node_a(_EMPTY_STATE) == {"captured_figures": [PNG]}
    assert node_b(_EMPTY_STATE) == {"captured_figures": [HTML]}
