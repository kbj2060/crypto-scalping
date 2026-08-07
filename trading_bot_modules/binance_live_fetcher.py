import asyncio
import logging
import math
import os

import pandas as pd
import ccxt.async_support as ccxt
from dotenv import load_dotenv

from trading_bot_modules.binance_runtime_config import BinanceAccountConfig

load_dotenv()

logger = logging.getLogger("LiveBot")


class Colors:
    GREEN, RED, YELLOW, CYAN, BLUE, MAGENTA, DIM, RESET, BOLD = (
        '\033[92m', '\033[91m', '\033[93m', '\033[96m',
        '\033[94m', '\033[95m', '\033[2m', '\033[0m', '\033[1m',
    )


def _safe_float(v, d: float = 0.0) -> float:
    try:
        x = float(v)
        return x if math.isfinite(x) else float(d)
    except Exception:
        return float(d)


# ════════════════════════════════════════════════════════════════
# 1. 데이터 수집기
# ════════════════════════════════════════════════════════════════
class BinanceLiveFetcher:
    def __init__(
        self,
        symbol='ETHUSDT',
        timeframe='5m',
        limit=2500,
        account_symbol=None,
        account_config: BinanceAccountConfig | None = None,
        account_api_key: str | None = None,
        account_api_secret: str | None = None,
    ):
        account_config = account_config or BinanceAccountConfig.from_env()
        self.symbol = symbol.replace('/', '')
        self.timeframe = timeframe
        self.ancillary_period = os.getenv("BINANCE_ANCILLARY_PERIOD", "5m")
        self.limit = limit
        self.exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'future'},
        })
        self.account_enabled = bool(account_config.enabled)
        self.account_position_sync_enabled = bool(account_config.position_sync_enabled)
        self.account_testnet = bool(account_config.testnet)
        # `account_symbol` constructor arg takes priority over the process-wide
        # BINANCE_ACCOUNT_SYMBOL env var so multiple fetchers (one per asset) in the same
        # process don't all collapse onto whichever single symbol that env var names.
        # Omitting it (the default) preserves the exact prior behavior for existing callers.
        self.account_symbol = str(account_symbol or account_config.symbol or self._to_unified_perp_symbol(self.symbol))
        # `account_api_key`/`account_api_secret` let a caller point this fetcher at a DIFFERENT
        # Binance account than the process-wide BINANCE_API_KEY/BINANCE_SECRET_KEY env vars (e.g.
        # a separate dedicated sub-account for a different strategy). Omitting them (the default,
        # None) preserves the exact prior behavior -- credentials come from the global env vars,
        # unchanged for every existing caller.
        self._account_api_key_override = account_api_key
        self._account_api_secret_override = account_api_secret
        self.account_exchange = self._build_account_exchange()
        self.api_retries = int(os.getenv("BINANCE_API_RETRIES", "4"))
        self.api_retry_delay_sec = float(os.getenv("BINANCE_API_RETRY_DELAY_SEC", "1.5"))

    @staticmethod
    def _to_unified_perp_symbol(symbol: str) -> str:
        s = str(symbol or "").replace("/", "").replace(":USDT", "").upper()
        if s.endswith("USDT") and len(s) > 4:
            return f"{s[:-4]}/USDT:USDT"
        return str(symbol or "")

    def _build_account_exchange(self):
        if not self.account_enabled:
            return None
        api_key = (self._account_api_key_override or os.getenv("BINANCE_API_KEY", "")).strip()
        secret = (
            self._account_api_secret_override
            or os.getenv("BINANCE_SECRET_KEY", "")
            or os.getenv("BINANCE_API_SECRET", "")
            or os.getenv("BINANCE_SECRET", "")
        ).strip()
        if not api_key or not secret:
            logger.warning("SYSTEM binance_account=OFF reason=missing_api_key_or_secret")
            self.account_enabled = False
            self.account_position_sync_enabled = False
            return None
        ex = ccxt.binance({
            'apiKey': api_key,
            'secret': secret,
            'enableRateLimit': True,
            'options': {'defaultType': 'future'},
        })
        if self.account_testnet:
            # IMPORTANT: ccxt has deprecated Binance futures sandbox/testnet support
            # (https://t.me/ccxt_announcements/92). `set_sandbox_mode(True)` itself no longer
            # raises -- it silently succeeds while leaving the exchange pointed at REAL mainnet
            # endpoints. The failure only surfaces later, the first time an authenticated futures
            # call (fetch_balance/fetch_positions/create_order) actually runs -- by which point a
            # caller who believed account_testnet=True meant "safely isolated from real funds"
            # would already be holding a live-mainnet-connected exchange object. Verified directly
            # against ccxt: `set_sandbox_mode(True)` returns cleanly, `fetch_balance` afterward
            # raises `NotSupported`. Because there is currently no working way to get an isolated
            # Binance futures testnet connection through ccxt, treat requesting testnet mode as an
            # unconditional refusal to build a real account connection at all -- fail closed rather
            # than silently trading real funds on mainnet while believing it's a test.
            logger.warning(
                "SYSTEM binance_account testnet=requested but ccxt no longer supports Binance "
                "futures sandbox mode (set_sandbox_mode succeeds without isolating anything) -- "
                "refusing to build a real account connection rather than risk silently running "
                "against live mainnet; disabling account features"
            )
            self.account_enabled = False
            self.account_position_sync_enabled = False
            return None
        logger.info(
            "SYSTEM binance_account=ON testnet=%s position_sync=%s symbol=%s",
            self.account_testnet,
            self.account_position_sync_enabled,
            self.account_symbol,
        )
        return ex

    def account_status(self) -> dict:
        ready = bool(self.account_enabled and self.account_exchange is not None)
        return {
            "enabled": bool(self.account_enabled),
            "ready": ready,
            "testnet": bool(self.account_testnet),
            "position_sync_configured": bool(self.account_position_sync_enabled),
            "position_sync_enabled": bool(ready and self.account_position_sync_enabled),
            "symbol": str(self.account_symbol),
        }

    async def fetch_account_position(self) -> dict | None:
        if not self.account_enabled or self.account_exchange is None or not self.account_position_sync_enabled:
            self._last_position_query_ok = False
            self._last_position_query_error = "position_sync_not_ready"
            return None
        try:
            positions = await self._call_with_retry(
                f"fetch_account_position[{self.account_symbol}]",
                lambda: self.account_exchange.fetch_positions([self.account_symbol]),
            )
            self._last_position_query_ok = True
            self._last_position_query_error = ""
            for p in positions or []:
                info = dict(p.get("info", {}) or {})
                contracts = _safe_float(p.get("contracts", info.get("positionAmt", 0.0)), 0.0)
                position_amt = _safe_float(info.get("positionAmt", contracts), contracts)
                size = position_amt if abs(position_amt) >= abs(contracts) else contracts
                if abs(size) <= 1e-12:
                    continue
                side = str(p.get("side") or "").upper()
                if side not in {"LONG", "SHORT"}:
                    side = "LONG" if size > 0 else "SHORT"
                entry = _safe_float(
                    p.get("entryPrice", p.get("entry_price", info.get("entryPrice", 0.0))),
                    0.0,
                )
                leverage = _safe_float(p.get("leverage", info.get("leverage", 0.0)), 0.0)
                notional = abs(_safe_float(p.get("notional", info.get("notional", 0.0)), 0.0))
                return {
                    "type": side,
                    "entry_price": float(entry),
                    "leverage": float(leverage),
                    "contracts": float(abs(size)),
                    "notional": float(notional),
                    "source": "binance_account",
                    "testnet": bool(self.account_testnet),
                }
        except Exception as e:
            self._last_position_query_ok = False
            self._last_position_query_error = str(e)
            logger.warning("SYSTEM binance_account position=BAD reason=%s", e)
        return None

    async def fetch_account_snapshot(self) -> dict:
        status = self.account_status()
        out = {
            "status": status,
            "balance_ok": False,
            "position": None,
            "position_query_ok": False,
            "position_query_error": "position_sync_not_ready" if not status["position_sync_enabled"] else "",
            "error": "",
        }
        if not status["ready"]:
            return out
        try:
            bal = await self._call_with_retry(
                "fetch_account_balance[futures]",
                lambda: self.account_exchange.fetch_balance(params={"type": "future"}),
            )
            total = dict((bal or {}).get("total", {}) or {})
            free = dict((bal or {}).get("free", {}) or {})
            out["balance_ok"] = True
            # Real execution is USDT-M futures (see trading_bot.py OMEGA4_6_1_SHADOW_ASSET_CONFIG
            # and .env BINANCE_ACCOUNT_SYMBOL) -- account balance settles in USDT.
            out["balance"] = {
                "USDT_total": _safe_float(total.get("USDT", 0.0), 0.0),
                "USDT_free": _safe_float(free.get("USDT", 0.0), 0.0),
            }
        except Exception as e:
            out["error"] = str(e)
            logger.warning("SYSTEM binance_account balance=BAD reason=%s", e)
        out["position"] = await self.fetch_account_position()
        out["position_query_ok"] = bool(getattr(self, "_last_position_query_ok", False))
        out["position_query_error"] = str(getattr(self, "_last_position_query_error", "") or "")
        if isinstance(out.get("position"), dict):
            equity = _safe_float(dict(out.get("balance", {}) or {}).get("USDT_total", 0.0), 0.0)
            notional = _safe_float(out["position"].get("notional", 0.0), 0.0)
            out["position"]["account_equity_usdt"] = float(equity)
            if equity > 0.0 and notional > 0.0:
                out["position"]["notional_exposure"] = float(notional / equity)
        return out

    async def close(self) -> None:
        await self.exchange.close()
        if self.account_exchange is not None:
            await self.account_exchange.close()



    async def _call_with_retry(self, label: str, fn):
        last_error = None
        for attempt in range(1, self.api_retries + 1):
            try:
                return await fn()
            except Exception as e:
                last_error = e
                if attempt >= self.api_retries:
                    break
                sleep_sec = self.api_retry_delay_sec * attempt
                logger.warning(
                    "⚠️ %s 실패(%d/%d): %s | %.1fs 후 재시도",
                    label,
                    attempt,
                    self.api_retries,
                    e,
                    sleep_sec,
                )
                await asyncio.sleep(sleep_sec)
        raise RuntimeError(f"{label} failed after {self.api_retries} attempts") from last_error

    def load_local_data(self):
        try:
            eth_df = pd.read_csv('data/test/eth_test_data.csv')
            btc_df = pd.read_csv('data/test/btc_test_data.csv')
            for df in [eth_df, btc_df]:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                cols = df.columns.drop('timestamp')
                df[cols] = df[cols].apply(pd.to_numeric, errors='raise')
            logger.info(f"{Colors.GREEN}📂 로컬 데이터 로드 성공{Colors.RESET}")
            return eth_df, btc_df
        except Exception as e:
            logger.error(f"로컬 로드 실패: {e}")
            return None, None

    async def fetch_klines_raw(self, symbol, target_limit):
        all_klines = []
        last_end_time = None
        while len(all_klines) < target_limit:
            params = {'symbol': symbol, 'interval': self.timeframe, 'limit': 1000}
            if last_end_time is not None:
                params['endTime'] = int(last_end_time) - 1
            klines = await self._call_with_retry(
                f"fetch_klines_raw[{symbol}]",
                lambda: self.exchange.fapiPublicGetKlines(params),
            )
            if not klines: break
            all_klines = klines + all_klines
            last_end_time = int(klines[0][0])
            if len(klines) < 1000: break
        return all_klines[-target_limit:]

    async def fetch_ancillary_data(self, limit=500):
        tasks = [
            self.exchange.fapiDataGetOpenInterestHist({'symbol': self.symbol, 'period': self.ancillary_period, 'limit': limit}),
            self.exchange.fapiDataGetTopLongShortAccountRatio({'symbol': self.symbol, 'period': self.ancillary_period, 'limit': limit}),
            self.exchange.fapiDataGetTopLongShortPositionRatio({'symbol': self.symbol, 'period': self.ancillary_period, 'limit': limit}),
            self.exchange.fapiDataGetGlobalLongShortAccountRatio({'symbol': self.symbol, 'period': self.ancillary_period, 'limit': limit}),
            self.exchange.fapiDataGetTakerlongshortRatio({'symbol': self.symbol, 'period': self.ancillary_period, 'limit': limit}),
            self.exchange.fapiPublicGetFundingRate({'symbol': self.symbol, 'limit': limit})
        ]
        return await asyncio.gather(*tasks, return_exceptions=True)

    def _process_to_df(self, eth_klines, btc_klines, ancillary_results):
        eth_df = pd.DataFrame(eth_klines).iloc[:, :11]
        eth_df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote']
        eth_df['timestamp'] = pd.to_numeric(eth_df['timestamp'], errors='coerce')
        eth_df['timestamp'] = pd.to_datetime(eth_df['timestamp'], unit='ms')
        eth_df[eth_df.columns.drop('timestamp')] = eth_df[eth_df.columns.drop('timestamp')].apply(pd.to_numeric, errors='raise')

        btc_df = pd.DataFrame(btc_klines).iloc[:, [0, 4, 5, 7]]
        btc_df.columns = ['timestamp', 'close_btc', 'volume_btc', 'quote_volume_btc']
        btc_df['timestamp'] = pd.to_numeric(btc_df['timestamp'], errors='coerce')
        btc_df['timestamp'] = pd.to_datetime(btc_df['timestamp'], unit='ms')
        btc_df[btc_df.columns.drop('timestamp')] = btc_df[btc_df.columns.drop('timestamp')].apply(pd.to_numeric, errors='raise')

        if ancillary_results:
            mappings = [
                (0, 'sumOpenInterestValue', 'sum_open_interest_value'),
                (1, 'longShortRatio', 'sum_toptrader_long_short_ratio'),
                (2, 'longShortRatio', 'count_toptrader_long_short_ratio'),
                (3, 'longShortRatio', 'count_long_short_ratio'),
                (4, 'buySellRatio', 'taker_long_short_ratio'),
                (5, 'fundingRate', 'last_funding_rate'),
            ]
            for idx, key, new_name in mappings:
                res = ancillary_results[idx]
                if isinstance(res, Exception):
                    raise RuntimeError(f"ancillary[{idx}] fetch failed for {new_name}: {res}") from res
                if isinstance(res, list) and len(res) > 0:
                    try:
                        temp_df = pd.DataFrame(res)
                        t_col = next((c for c in ['timestamp', 'fundingTime', 'time'] if c in temp_df.columns), None)
                        if t_col and key in temp_df.columns:
                            subset = temp_df[[t_col, key]].rename(columns={t_col: 'timestamp', key: new_name})
                            subset['timestamp'] = pd.to_numeric(subset['timestamp'], errors='coerce')
                            subset['timestamp'] = pd.to_datetime(subset['timestamp'], unit='ms')
                            subset[new_name] = pd.to_numeric(subset[new_name], errors='raise')
                            eth_df = pd.merge_asof(
                                eth_df.sort_values('timestamp'),
                                subset.sort_values('timestamp'),
                                on='timestamp',
                                direction='backward',
                            )
                    except Exception: raise
        required = [
            'sum_open_interest_value',
            'sum_toptrader_long_short_ratio',
            'count_toptrader_long_short_ratio',
            'count_long_short_ratio',
            'taker_long_short_ratio',
            'last_funding_rate',
        ]
        missing = [c for c in required if c not in eth_df.columns]
        if missing:
            raise RuntimeError(f"ancillary columns missing after merge: {','.join(missing)}")
        eth_df = eth_df.sort_values('timestamp').reset_index(drop=True)
        btc_df = btc_df.sort_values('timestamp').reset_index(drop=True)
        eth_df = eth_df.ffill()
        ready = eth_df[required].notna().all(axis=1)
        if not ready.all():
            if not ready.any():
                raise RuntimeError("No causally complete ancillary snapshot exists in live fetch window")
            first_ready_pos = int(ready[ready].index[0])
            first_ready_ts = eth_df.loc[first_ready_pos, 'timestamp']
            logger.warning(
                "SYSTEM live_causal_trim leading_rows=%d first_complete_ts=%s reason=incomplete_ancillary_warmup",
                first_ready_pos,
                first_ready_ts,
            )
            eth_df = eth_df.iloc[first_ready_pos:].reset_index(drop=True)
            btc_df = btc_df[btc_df['timestamp'] >= first_ready_ts].reset_index(drop=True)
        nan_cols = [c for c in eth_df.columns if eth_df[c].isna().any()]
        if nan_cols:
            raise RuntimeError(f"NaN values remain after causal ffill in columns: {', '.join(nan_cols)}")
        return eth_df, btc_df

    async def fetch_initial_data(self):
        eth_klines = await self.fetch_klines_raw(self.symbol, self.limit)
        btc_klines = await self.fetch_klines_raw('BTCUSDT', self.limit)
        ancillary = await self.fetch_ancillary_data(500)
        return self._process_to_df(eth_klines, btc_klines, ancillary)

    async def fetch_latest_patch(self):
        eth_klines = await self._call_with_retry(
            f"fetch_latest_patch[{self.symbol}]",
            lambda: self.exchange.fapiPublicGetKlines({'symbol': self.symbol, 'interval': self.timeframe, 'limit': 5}),
        )
        btc_klines = await self._call_with_retry(
            "fetch_latest_patch[BTCUSDT]",
            lambda: self.exchange.fapiPublicGetKlines({'symbol': 'BTCUSDT', 'interval': self.timeframe, 'limit': 5}),
        )
        ancillary = await self.fetch_ancillary_data(5)
        return self._process_to_df(eth_klines, btc_klines, ancillary)
