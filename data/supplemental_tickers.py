"""
Supplemental tickers for swing screening.

Popular/liquid tickers NOT in the S&P 500 but commonly traded for swings.
Easy to update — just add or remove from the list.
"""
from __future__ import annotations

SUPPLEMENTAL_TICKERS: list[str] = [
    # EV / Clean Energy
    "RIVN", "LCID", "NIO", "XPEV", "LI",
    # Fintech / Crypto-adjacent
    "COIN", "SOFI", "HOOD", "MARA", "RIOT", "CLSK", "HUT",
    # Tech / Growth
    "PLTR", "RBLX", "SNAP", "U", "DKNG", "SE", "PINS",
    "ROKU", "SHOP", "SQ", "TWLO", "NET", "CRWD", "SNOW",
    # Biotech / Pharma
    "MRNA", "BNTX", "DNA",
    # Consumer / Meme-popular
    "GME", "AMC", "BBBY", "WISH",
    # Industrials / Space
    "JOBY", "RKLB", "LUNR",
    # Semiconductors
    "ARM", "SMCI",
    # Cannabis
    "TLRY", "CGC",
]
