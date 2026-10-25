import io
import unittest

from training.run_train import Tee


class WindowsEncodingStream:
    encoding = "cp1252"

    def __init__(self):
        self.value = ""

    def write(self, text):
        self.value += text.encode(self.encoding, errors="strict").decode(self.encoding)

    def flush(self):
        pass


class TrainingRunnerTests(unittest.TestCase):
    def test_tee_preserves_utf8_log_when_terminal_cannot_encode_currency(self):
        terminal = WindowsEncodingStream()
        log = io.StringIO()

        Tee(terminal, log).write("P&L: ₹100\n")

        self.assertEqual(terminal.value, "P&L: ?100\n")
        self.assertEqual(log.getvalue(), "P&L: ₹100\n")


if __name__ == "__main__":
    unittest.main()
