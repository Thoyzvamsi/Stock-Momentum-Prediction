import unittest

from ticker_utils import normalize_ticker_symbol


class TickerSymbolTests(unittest.TestCase):
    def test_adds_nse_suffix_to_bare_symbol(self):
        self.assertEqual(normalize_ticker_symbol("HDFCBANK"), "HDFCBANK.NS")

    def test_trims_and_uppercases_before_adding_suffix(self):
        self.assertEqual(normalize_ticker_symbol("  reliance "), "RELIANCE.NS")

    def test_preserves_explicit_exchange_suffix(self):
        self.assertEqual(normalize_ticker_symbol("hdfcbank.bo"), "HDFCBANK.BO")

    def test_preserves_symbols_with_dot_and_index_symbols(self):
        self.assertEqual(normalize_ticker_symbol("brk.b"), "BRK.B")
        self.assertEqual(normalize_ticker_symbol("^NSEI"), "^NSEI")

    def test_empty_input_stays_empty(self):
        self.assertEqual(normalize_ticker_symbol("  "), "")


if __name__ == "__main__":
    unittest.main()
