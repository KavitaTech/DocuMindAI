import unittest
from rag_core import chunk_text, build_context


class RAGCoreTests(unittest.TestCase):

    def test_empty_text(self):
        self.assertEqual(chunk_text(""), [])

    def test_small_text(self):
        self.assertEqual(
            chunk_text("Hello document assistant"),
            ["Hello document assistant"]
        )

    def test_chunk_overlap(self):
        text = " ".join(str(i) for i in range(12))
        result = chunk_text(text, words_per_chunk=5, overlap=2)

        self.assertEqual(
            result[0].split()[-2:],
            result[1].split()[:2]
        )

    def test_invalid_chunking(self):
        with self.assertRaises(ValueError):
            chunk_text("Hello", words_per_chunk=5, overlap=5)

    def test_source_context(self):
        result = build_context([{
            "text": "Annual leave entitlement is 18 days.",
            "metadata": {
                "filename": "Handbook.txt",
                "page": 2
            }
        }])

        self.assertIn("[S1]", result)
        self.assertIn("Handbook.txt", result)
        self.assertIn("Page: 2", result)


if __name__ == "__main__":
    unittest.main()
