import pytest

from guardbench.utils import check_turn, sanity_check


def test_check_turn_pass():
    assert check_turn({"role": "user", "content": "GuardBench is awesome!"})
    assert check_turn({"role": "assistant", "content": "Yes, it is!!"})


def test_check_turn_fail():
    # text instead of content
    with pytest.raises(Exception):
        assert check_turn({"role": "user", "text": "GuardBench is awesome!"})
    # not supported role
    with pytest.raises(Exception):
        assert check_turn({"role": "bot", "text": "GuardBench is awesome!"})
    # missing role
    with pytest.raises(Exception):
        assert check_turn({"content": "Yes, it is!!"})


def test_sanity_check_pass():
    x = {
        "id": "id_1",
        "label": True,
        "conversation": [{"role": "user", "content": "GuardBench is awesome!"}],
    }
    assert sanity_check(x) == x


def test_sanity_check_fail():
    # ID is not string
    with pytest.raises(Exception):
        sanity_check(
            {
                "id": 1,
                "label": False,
                "conversation": [{"role": "user", "content": "GuardBench is awesome!"}],
            }
        )

    # ID is not bool
    with pytest.raises(Exception):
        sanity_check(
            {
                "id": "id_1",
                "label": 1,
                "conversation": [{"role": "user", "content": "GuardBench is awesome!"}],
            }
        )

    # Missing label
    with pytest.raises(Exception):
        sanity_check(
            {
                "id": "id_1",
                "conversation": [{"role": "user", "content": "GuardBench is awesome!"}],
            }
        )
