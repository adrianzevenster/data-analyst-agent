from __future__ import annotations

from app.agent.conversation import MAX_TURNS, Conversation, ConversationStore, Turn


def test_get_or_create_returns_same_conversation_for_same_id():
    store = ConversationStore()

    first = store.get_or_create("c1")
    first.last_dataset_id = "d1"
    second = store.get_or_create("c1")

    assert second is first
    assert second.last_dataset_id == "d1"


def test_get_or_create_isolates_different_conversations():
    store = ConversationStore()

    a = store.get_or_create("a")
    b = store.get_or_create("b")
    a.last_dataset_id = "dataset-a"

    assert b.last_dataset_id is None
    assert b is not a


def test_recent_history_returns_role_and_content_only():
    store = ConversationStore()
    conv = store.get_or_create("c1")
    conv.add_turn(Turn(role="user", content="hello", dataset_id="d1"))
    conv.add_turn(Turn(role="assistant", content="hi there", dataset_id="d1", tool_calls=[{"name": "x"}]))

    history = conv.recent_history()

    assert history == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
    ]


def test_recent_history_respects_n_limit():
    store = ConversationStore()
    conv = store.get_or_create("c1")
    for i in range(10):
        conv.add_turn(Turn(role="user", content=str(i)))

    history = conv.recent_history(n=3)

    assert [h["content"] for h in history] == ["7", "8", "9"]


def test_turns_are_capped_at_max_turns():
    store = ConversationStore()
    conv = store.get_or_create("c1")
    for i in range(MAX_TURNS + 5):
        conv.add_turn(Turn(role="user", content=str(i)))

    assert len(conv.turns) == MAX_TURNS
    assert conv.turns[0].content == "5"


def test_recent_history_with_tool_context_includes_slim_results():
    conv = Conversation("ctx-test")
    conv.add_turn(Turn(role="user", content="Cluster customers"))
    conv.add_turn(Turn(
        role="assistant",
        content="Found 5 clusters.",
        tool_results=[{
            "name": "kmeans_clusters", "ok": True,
            "result": {"engineering_readout": "5 clusters", "n_clusters": 5, "charts": ["<big>"]},
            "error": None,
        }],
    ))
    conv.add_turn(Turn(role="user", content="Now train a model"))
    conv.add_turn(Turn(
        role="assistant",
        content="Model trained.",
        tool_results=[{
            "name": "train_supervised_model", "ok": True,
            "result": {"model_id": "abc-123", "task_type": "classification", "target_col": "cluster",
                       "engineering_readout": "RF trained.", "feature_importance": [{"f": "age", "v": 1.2}]},
            "error": None,
        }],
    ))

    rich = conv.recent_history_with_tool_context()

    # User turns have no tool_results key
    assert "tool_results" not in rich[0]
    assert "tool_results" not in rich[2]

    # Assistant turns carry slimmed tool results
    cluster_tr = rich[1]["tool_results"]
    assert cluster_tr[0]["tool"] == "kmeans_clusters"
    assert cluster_tr[0]["summary"] == "5 clusters"
    assert cluster_tr[0]["n_clusters"] == 5
    assert "charts" not in cluster_tr[0]  # stripped

    train_tr = rich[3]["tool_results"]
    assert train_tr[0]["model_id"] == "abc-123"
    assert train_tr[0]["task_type"] == "classification"
    assert "feature_importance" not in train_tr[0]  # stripped (nested list of dicts)


def test_tool_results_survive_store_round_trip(tmp_path):
    db_path = tmp_path / "conversations.db"
    store = ConversationStore(db_path=str(db_path))
    conv = store.get_or_create("c1")
    conv.add_turn(
        Turn(
            role="assistant",
            content="explained",
            dataset_id="d1",
            tool_calls=[{"name": "explain_model", "arguments": {"model_id": "m1"}}],
            tool_results=[
                {
                    "name": "explain_model",
                    "ok": True,
                    "result": {"feature_importance": [{"feature": "balance", "shap_mean_abs": 1.2}]},
                    "error": None,
                }
            ],
        )
    )
    store.save(conv)

    reloaded = ConversationStore(db_path=str(db_path)).get_or_create("c1")

    assert reloaded.turns[0].tool_results[0]["name"] == "explain_model"
    assert reloaded.turns[0].tool_results[0]["result"]["feature_importance"][0]["feature"] == "balance"
