"""계좌 카드 날짜별 메모 저장(dashboard/acct_notes.py). python -m pytest -q test/test_acct_notes_20261104.py"""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dashboard import acct_notes  # noqa: E402


class AcctNotes(unittest.TestCase):
    def setUp(self):
        self.p = Path(tempfile.mkdtemp()) / "sub" / "n.json"

    def test_roundtrip_and_clear(self):
        self.assertEqual(acct_notes.load(self.p), {})   # 파일 없음
        self.assertEqual(acct_notes.save(self.p, "2026-11-04", "손절 늦음", 5), {"ok": True, "at": 5})
        acct_notes.save(self.p, "2026-11-03", "굿", 6)
        self.assertEqual(acct_notes.load(self.p)["2026-11-04"], {"text": "손절 늦음", "at": 5})
        acct_notes.save(self.p, "2026-11-04", "수정", 7)   # 같은 날 덮어쓰기
        self.assertEqual(acct_notes.load(self.p)["2026-11-04"]["text"], "수정")
        self.assertIsNone(acct_notes.save(self.p, "2026-11-04", "  \n", 8)["at"])   # 빈 글 = 삭제
        self.assertEqual(list(acct_notes.load(self.p)), ["2026-11-03"])

    def test_rejects_bad_input(self):
        for date, text in (("11/04", "x"), ("2026-11-04; rm", "x"), (None, "x"), ("2026-11-04", None), ("2026-11-04", "가" * (acct_notes.NOTE_MAX + 1))):
            with self.assertRaises(ValueError):
                acct_notes.save(self.p, date, text)
        self.assertEqual(acct_notes.load(self.p), {})   # 거절하면 아무것도 안 쓴다

    def test_corrupt_file_is_empty(self):
        self.p.parent.mkdir(parents=True)
        self.p.write_text("{깨짐", encoding="utf-8")
        self.assertEqual(acct_notes.load(self.p), {})
        acct_notes.save(self.p, "2026-11-04", "복구", 1)
        self.assertEqual(acct_notes.load(self.p)["2026-11-04"]["text"], "복구")


    def test_wrong_notebook(self):
        r1 = acct_notes.wrong_add(self.p, "2026-11-05", "추격  진입\n물탐", "원칙 위반", 5)
        r2 = acct_notes.wrong_add(self.p, "2026-11-05", "가짜 돌파", "", 5)   # 같은 ms 라도 id 가 겹치지 않는다
        self.assertEqual(r1["text"], "추격 진입 물탐")
        self.assertNotEqual(r1["id"], r2["id"])
        acct_notes.wrong_del(self.p, r1["id"])
        self.assertEqual([r["text"] for r in acct_notes.wrong_load(self.p)], ["가짜 돌파"])
        for d, x, g in (("11/05", "x", ""), ("2026-11-05", "  ", ""), ("2026-11-05", "x", "없는태그"), ("2026-11-05", "가" * 301, "")):
            with self.assertRaises(ValueError):
                acct_notes.wrong_add(self.p, d, x, g)
        self.assertEqual(len(acct_notes.wrong_load(self.p)), 1)


if __name__ == "__main__":
    unittest.main()
