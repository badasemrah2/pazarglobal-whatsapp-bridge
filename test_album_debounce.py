"""
Album debounce regression tests.

A four-photo car used to produce four replies, because WhatsApp delivers each photo of an
album as its own webhook. These tests cover the two mechanisms that collapse it to one:
the ticket (who answers) and the pile (what they answer with).

Run: python test_album_debounce.py
"""
import sys

# main is imported first on purpose: it configures logging before pulling in
# redis_helper, which reports its connection state as it loads. Importing redis_helper
# first would reproduce the pre-fix ordering, where that line fell to logging's
# unformatted last-resort handler.
import main  # noqa: F401
import redis_helper
from redis_helper import redis_client


def reset_store():
    redis_helper._IN_MEMORY_STORE.clear()


# ── Ticket: exactly one webhook answers ──────────────────────────────────────

def test_only_the_last_ticket_survives():
    """Four concurrent photos, one reply."""
    reset_store()
    phone = "+905551112233"
    key = f"album_seq:{phone}"

    tickets = [redis_client.counter(key) for _ in range(4)]
    assert tickets == [1, 2, 3, 4], tickets

    final = redis_client.counter_value(key)
    survivors = [t for t in tickets if t == final]
    assert survivors == [4], f"expected one survivor, got {survivors}"


def test_single_photo_answers_immediately():
    """A lone photo must not be silenced by its own ticket."""
    reset_store()
    key = "album_seq:+905551112233"
    ticket = redis_client.counter(key)
    assert redis_client.counter_value(key) == ticket


def test_second_album_does_not_silence_itself():
    """A later album keeps counting up; its own last photo still wins."""
    reset_store()
    key = "album_seq:+905551112233"

    first = [redis_client.counter(key) for _ in range(4)]
    second = [redis_client.counter(key) for _ in range(2)]

    assert second == [5, 6], second
    final = redis_client.counter_value(key)
    assert final == 6
    assert [t for t in first + second if t == final] == [6]


def test_separate_phones_do_not_interfere():
    """Two sellers uploading at once must not silence each other."""
    reset_store()
    a_key, b_key = "album_seq:+905551112233", "album_seq:+905559998877"

    a1 = redis_client.counter(a_key)
    b1 = redis_client.counter(b_key)
    b2 = redis_client.counter(b_key)

    assert redis_client.counter_value(a_key) == a1, "A's only photo must still answer"
    assert redis_client.counter_value(b_key) == b2
    assert b1 != redis_client.counter_value(b_key)


def test_blocked_last_photo_lets_the_album_still_answer():
    """A rejected final photo must not take the whole album's reply down with it."""
    reset_store()
    key = "album_seq:+905551112233"

    good = [redis_client.counter(key) for _ in range(3)]
    blocked = redis_client.counter(key)
    assert blocked == 4

    # The blocked photo replies on its own and hands its ticket back.
    redis_client.counter_release(key)

    final = redis_client.counter_value(key)
    assert [t for t in good if t == final] == [3], f"photo 3 should answer, final={final}"


def test_failed_photo_keeps_its_ticket():
    """A photo that fails to download must still be able to answer for the album.

    It holds no photos of its own, but its siblings' are in the pile - and if it gave the
    ticket back, an album where every photo failed would release every ticket and leave
    nobody to reply at all.
    """
    reset_store()
    key = "album_seq:+905551112233"
    draft = "draft-allfail"
    from main import album_collect

    tickets = [redis_client.counter(key) for _ in range(3)]
    for _ in tickets:
        redis_client.counter(f"album_failed:{draft}")

    final = redis_client.counter_value(key)
    assert [t for t in tickets if t == final] == [3], "photo 3 must still answer"
    assert redis_client.counter_value(f"album_failed:{draft}") == 3

    # Nothing was gathered, so the survivor keeps the text it already has.
    paths, _, body = album_collect(draft, [], [], "2011 model Jetta")
    assert paths == []
    assert body == "2011 model Jetta"


def test_survivor_answers_with_siblings_photos_when_its_own_failed():
    """One photo 404s at Twilio; the reply still carries the three that arrived."""
    reset_store()
    draft = "draft-partial"
    from main import album_contribute, album_collect

    album_contribute(draft, ["p/1.jpg"], [{"product": "Jetta"}], "2011 model Jetta")
    album_contribute(draft, ["p/2.jpg"], [{"product": "Jetta arka"}], "")
    album_contribute(draft, ["p/3.jpg"], [{"product": "Jetta iç"}], "")
    redis_client.counter(f"album_failed:{draft}")  # the fourth never downloaded

    # The failed webhook is last, so it answers - with nothing of its own to add.
    paths, vision, body = album_collect(draft, [], [], "")

    assert paths == ["p/1.jpg", "p/2.jpg", "p/3.jpg"], paths
    assert len(vision) == 3
    assert body == "2011 model Jetta"
    assert redis_client.counter_value(f"album_failed:{draft}") == 1


def test_every_photo_blocked_leaves_no_survivor():
    """Nothing uploaded means nothing to summarise; only the block messages go out."""
    reset_store()
    key = "album_seq:+905551112233"

    tickets = [redis_client.counter(key) for _ in range(3)]
    for _ in tickets:
        redis_client.counter_release(key)

    assert redis_client.counter_value(key) == 0
    assert [t for t in tickets if t == 0] == []


# ── Pile: the survivor answers with the whole album ──────────────────────────

def test_collect_merges_all_four_photos():
    reset_store()
    draft = "draft-abc"
    from main import album_contribute, album_collect

    album_contribute(draft, ["a/1.jpg"], [{"product": "Volkswagen Jetta"}], "2011 model Jetta, 180.000 km")
    album_contribute(draft, ["a/2.jpg"], [{"product": "Jetta iç mekan"}], "")
    album_contribute(draft, ["a/3.jpg"], [{"product": "Jetta arka"}], "")
    album_contribute(draft, ["a/4.jpg"], [{"product": "Jetta motor"}], "")

    paths, vision, body = album_collect(draft, ["a/4.jpg"], [{"product": "Jetta motor"}], "")

    assert paths == ["a/1.jpg", "a/2.jpg", "a/3.jpg", "a/4.jpg"], paths
    assert len(vision) == 4, vision
    # The description arrived with photo 1, which is not the webhook that replies.
    assert body == "2011 model Jetta, 180.000 km", body


def test_collect_preserves_turkish_characters():
    """Turkish text survives the JSON round-trip through Redis."""
    reset_store()
    draft = "draft-tr"
    from main import album_contribute, album_collect

    text = "İstanbul'da satılık, çok temiz, hasar kaydı yok. Şanzıman otomatik."
    album_contribute(draft, ["p/1.jpg"], [{"product": "Ütü", "condition": "İkinci el"}], text)

    _, vision, body = album_collect(draft, [], [], "")
    assert body == text, body
    assert vision[0]["product"] == "Ütü"
    assert vision[0]["condition"] == "İkinci el"


def test_collect_joins_multiple_captions():
    """A seller who types under two photos gets both lines, in order."""
    reset_store()
    draft = "draft-multi"
    from main import album_contribute, album_collect

    album_contribute(draft, ["p/1.jpg"], [], "2011 model Jetta")
    album_contribute(draft, ["p/2.jpg"], [], "Fiyat 450.000 TL")

    _, _, body = album_collect(draft, [], [], "")
    assert body == "2011 model Jetta\nFiyat 450.000 TL", repr(body)


def test_collect_falls_back_when_pile_is_empty():
    """A Redis outage must not blank out the message the webhook already has."""
    reset_store()
    from main import album_collect

    paths, vision, body = album_collect(
        "draft-missing", ["local/1.jpg"], [{"product": "Bisiklet"}], "Satılık bisiklet"
    )
    assert paths == ["local/1.jpg"]
    assert vision == [{"product": "Bisiklet"}]
    assert body == "Satılık bisiklet"


def test_collect_drains_the_pile():
    """A second album on the same draft must not re-send the first album's photos."""
    reset_store()
    draft = "draft-drain"
    from main import album_contribute, album_collect

    album_contribute(draft, ["p/1.jpg"], [], "ilk")
    album_collect(draft, [], [], "")

    album_contribute(draft, ["p/2.jpg"], [], "ikinci")
    paths, _, body = album_collect(draft, [], [], "")

    assert paths == ["p/2.jpg"], paths
    assert body == "ikinci", body


def test_media_only_message_keeps_empty_body():
    """No caption anywhere means no invented text for the agent to treat as a description."""
    reset_store()
    draft = "draft-nocaption"
    from main import album_contribute, album_collect

    album_contribute(draft, ["p/1.jpg"], [], "")
    album_contribute(draft, ["p/2.jpg"], [], "   ")

    _, _, body = album_collect(draft, [], [], "")
    assert body == "", repr(body)


# ── Long replies are split, not cut off ──────────────────────────────────────

def _search_reply(count: int) -> str:
    """A search result shaped the way search_agents builds one."""
    desc = "Temiz, bakımlı, hasar kaydı yok. Antep teslim. Detaylı bilgi için mesaj atabilirsiniz."
    blocks = [
        f"{i}. 20{10 + i} Marka Model {i}.6 Dizel Paket, {300 + i}.000 km - "
        f"{400 + i * 25}.000 TL - Otomotiv\n{desc}\n"
        f"Mesaj Gönder: https://pazarglobal.com/contact/a1b2c3d4e5f6g7h8i9j{i}"
        for i in range(1, count + 1)
    ]
    return (
        f"Aramanıza uygun {count} ilan buldum:\n\n"
        + "\n\n".join(blocks)
        + "\n\nDetay için: '1 nolu ilanın detayını göster' yazabilirsiniz."
    )


def test_short_reply_stays_one_message():
    from main import _split_for_whatsapp

    parts, dropped = _split_for_whatsapp("Merhaba, nasıl yardımcı olabilirim?", 1130, 3)
    assert parts == ["Merhaba, nasıl yardımcı olabilirim?"]
    assert dropped == 0


def test_long_search_reply_keeps_every_listing():
    """The whole point: five listings found must be five listings delivered."""
    import re
    from main import _split_for_whatsapp

    body = _search_reply(5)
    limit = 1130
    assert len(body) > limit, "fixture must actually exceed one message"

    parts, dropped = _split_for_whatsapp(body, limit, 3)

    assert dropped == 0
    assert len(parts) > 1
    assert all(len(p) <= limit for p in parts), [len(p) for p in parts]

    found = sum(len(re.findall(r"^\d+\. 20", p, re.MULTILINE)) for p in parts)
    assert found == 5, f"lost listings: {found} of 5"


def test_split_falls_on_listing_boundaries():
    """A message must never begin in the middle of a listing."""
    from main import _split_for_whatsapp

    parts, _ = _split_for_whatsapp(_search_reply(6), 1130, 3)
    for part in parts[1:]:
        assert not part.lstrip().startswith("Mesaj Gönder"), part[:60]
        assert not part.lstrip().startswith("Temiz, bakımlı"), part[:60]


def test_oversized_single_block_falls_back_to_lines():
    """One listing longer than a whole message still has to go out."""
    from main import _split_for_whatsapp

    block = "\n".join(f"satır {i} " + "x" * 60 for i in range(40))
    parts, dropped = _split_for_whatsapp(block, 300, 10)

    assert len(parts) > 1
    assert all(len(p) <= 300 for p in parts), [len(p) for p in parts]
    assert dropped == 0


def test_exceeding_the_part_cap_reports_what_was_dropped():
    """A cap that truncates silently reads as "that was everything"."""
    from main import _split_for_whatsapp

    parts, dropped = _split_for_whatsapp(_search_reply(30), 600, 2)

    assert len(parts) == 2
    assert dropped > 0, "dropped content must be reported, not swallowed"


def test_empty_reply_produces_no_parts():
    from main import _split_for_whatsapp

    assert _split_for_whatsapp("", 1130, 3) == ([], 0)
    assert _split_for_whatsapp("   \n  ", 1130, 3) == ([], 0)


# ── Log routing ──────────────────────────────────────────────────────────────

def test_routine_logs_go_to_stdout_and_problems_to_stderr():
    """Railway reads stderr as an error, so ordinary INFO must not land there.

    logging.basicConfig() defaults to stderr, which tagged every "Incoming WhatsApp
    message" as severity:"error" and left real failures with nothing to stand out
    against. This asserts the split rather than the config, so reintroducing
    basicConfig() anywhere would fail it.
    """
    import io as _io
    import logging
    import main  # noqa: F401  (importing configures logging)

    root = logging.getLogger()
    streams = {}
    for handler in root.handlers:
        buf = _io.StringIO()
        handler.stream = buf
        streams[handler] = buf

    log = logging.getLogger("main")
    log.info("routine line")
    log.warning("needs attention")
    log.error("broken")

    out = "".join(
        b.getvalue() for h, b in streams.items() if h.level < logging.WARNING
    )
    err = "".join(
        b.getvalue() for h, b in streams.items() if h.level >= logging.WARNING
    )

    assert "routine line" in out, out
    assert "routine line" not in err, err
    assert "needs attention" in err, err
    assert "broken" in err, err
    # A record printed on both streams would show up twice in Railway.
    assert "needs attention" not in out, out


def test_twilio_header_dumps_are_silenced_by_default():
    """A dozen header lines per sent message, saying nothing our own logs do not."""
    import logging
    import main  # noqa: F401

    assert logging.getLogger("twilio.http_client").level >= logging.WARNING
    assert logging.getLogger("httpx").level >= logging.WARNING


def run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = []
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            failures.append((t.__name__, e))
            print(f"  FAIL  {t.__name__}: {e}")
        except Exception as e:
            failures.append((t.__name__, e))
            print(f"  ERROR {t.__name__}: {type(e).__name__}: {e}")

    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(run())
