from memory_store import MemoryEntry, load_memory_db, recommend_memory_entries, upsert_memory_entries


def test_memory_db_upserts_and_recommends_top_entries(tmp_path):
    memory_path = tmp_path / "paper_memory.json"
    entries = [
        MemoryEntry(
            title="Agent Memory Paper",
            tags=["agent", "memory"],
            embedding=[1.0, 0.0],
            summary="Agent memory summary.",
            paper_type="Research",
            project_idea="Build an agent memory prototype.",
            link="Papers/agent-memory-paper",
            source_path="/vault/Papers/agent-memory-paper.md",
            note_type="Paper",
        ),
        MemoryEntry(
            title="Robotics Paper",
            tags=["robotics"],
            embedding=[0.0, 1.0],
            summary="Robotics summary.",
            paper_type="Research",
            project_idea="Build a robot demo.",
            link="Papers/robotics-paper",
            source_path="/vault/Papers/robotics-paper.md",
            note_type="Paper",
        ),
    ]

    upsert_memory_entries(memory_path, entries)
    loaded = load_memory_db(memory_path)
    recommendations = recommend_memory_entries(loaded, [1.0, 0.0], limit=1)

    assert len(loaded) == 2
    assert recommendations[0].title == "Agent Memory Paper"
    assert recommendations[0].link == "Papers/agent-memory-paper"
