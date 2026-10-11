from scripts.audit_release import MAX_BLOB_BYTES, audit


def test_reachable_git_history_is_publication_safe() -> None:
    result = audit()
    assert result["commits"] >= 6
    assert result["blobs"] > 0
    assert result["largest_blob_bytes"] < MAX_BLOB_BYTES
