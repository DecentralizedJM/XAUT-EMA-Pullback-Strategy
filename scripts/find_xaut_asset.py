#!/usr/bin/env python3
"""
Find XAUT asset on Mudrex using the official SDK.
https://github.com/DecentralizedJM/mudrex-api-trading-python-sdk

Usage: MUDREX_API_SECRET=your-secret python scripts/find_xaut_asset.py
"""
import os
import sys

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from mudrex import MudrexClient


def main():
    api_secret = os.getenv("MUDREX_API_SECRET")
    if not api_secret:
        print("Set MUDREX_API_SECRET in environment or .env")
        sys.exit(1)

    client = MudrexClient(api_secret=api_secret)

    # 1. Search for XAUT
    print("Searching assets for 'XAUT'...")
    try:
        assets = client.assets.search("XAUT")
        if assets:
            print(f"\nFound {len(assets)} asset(s) matching 'XAUT':")
            for a in assets:
                print(f"  {a}")
        else:
            print("No results from search. Trying list_all()...")
    except Exception as e:
        print(f"Search failed: {e}")
        assets = []

    # 2. Try get by symbol directly
    for sym in ["XAUTUSDT", "XAUT-USDT", "XAUT"]:
        print(f"\nTrying client.assets.get('{sym}')...")
        try:
            asset = client.assets.get(sym)
            print(f"  Found: {asset}")
            if hasattr(asset, "asset_id"):
                print(f"  asset_id: {asset.asset_id}")
            if hasattr(asset, "quantity_step"):
                print(f"  quantity_step: {asset.quantity_step}")
            if hasattr(asset, "min_quantity"):
                print(f"  min_quantity: {asset.min_quantity}")
            break
        except Exception as e:
            print(f"  Error: {e}")
    else:
        # 3. Fallback: scan full list
        print("\nScanning full asset list for XAUT...")
        try:
            all_assets = client.assets.list_all()
            xaut = [a for a in all_assets if "XAUT" in (getattr(a, "symbol", "") or "").upper()]
            if xaut:
                print(f"Found {len(xaut)} asset(s):")
                for a in xaut:
                    print(f"  {a}")
            else:
                print(f"Total assets: {len(all_assets)}. XAUT not found.")
        except Exception as e:
            print(f"list_all failed: {e}")


if __name__ == "__main__":
    main()
