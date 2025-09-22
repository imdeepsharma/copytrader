# Notes:
# - This script focuses on the common "swapExact*" functions of PancakeSwap V2.
# - It tries to mirror swaps with the same path, adjusting input size per config.
# - If you don't hold the source token, the trade is skipped (and logged).
# - Approvals are handled automatically for ERC-20 sources (non-BNB).
# - BSC chainId=56. We use legacy gasPrice for maximum compatibility.
# - Logs go to console and to ./copy_trader.log

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import signal
import sys
import time
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

from web3 import AsyncWeb3
from web3.providers.async_rpc import AsyncHTTPProvider
from web3.contract.async_contract import AsyncContract
from web3.types import TxData, TxReceipt

# Optional: load .env if present (development convenience)
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()  # nosec - dev only
except Exception:
    pass

# ----------------------------
# Constants & Minimal ABIs
# ----------------------------

# PancakeSwap V2 Router (MAINNET BSC)
ROUTER_V2_ADDRESS = "0x10ED43C718714eb63d5aA57B78B54704E256024E"

# WBNB (MAINNET BSC)
WBNB_ADDRESS = "0xBB4CdB9CBd36B01bD1cBaEBF2De08d9173bc095c"

# Common ERC-20 minimal ABI (balanceOf, decimals, symbol, allowance, approve)
ERC20_MIN_ABI: List[Dict[str, Any]] = [
    {
        "constant": True,
        "inputs": [{"name": "_owner", "type": "address"}],
        "name": "balanceOf",
        "outputs": [{"name": "balance", "type": "uint256"}],
        "type": "function",
        "stateMutability": "view",
    },
    {
        "constant": True,
        "inputs": [],
        "name": "decimals",
        "outputs": [{"name": "", "type": "uint8"}],
        "type": "function",
        "stateMutability": "view",
    },
    {
        "constant": True,
        "inputs": [],
        "name": "symbol",
        "outputs": [{"name": "", "type": "string"}],
        "type": "function",
        "stateMutability": "view",
    },
    {
        "constant": True,
        "inputs": [{"name": "_owner", "type": "address"}, {"name": "_spender", "type": "address"}],
        "name": "allowance",
        "outputs": [{"name": "remaining", "type": "uint256"}],
        "type": "function",
        "stateMutability": "view",
    },
    {
        "constant": False,
        "inputs": [{"name": "_spender", "type": "address"}, {"name": "_value", "type": "uint256"}],
        "name": "approve",
        "outputs": [{"name": "success", "type": "bool"}],
        "type": "function",
        "stateMutability": "nonpayable",
    },
]

# PancakeSwap V2 Router minimal ABI (swapExact variants + getAmountsOut)
ROUTER_V2_MIN_ABI: List[Dict[str, Any]] = [
    {
        "name": "getAmountsOut",
        "type": "function",
        "stateMutability": "view",
        "inputs": [{"name": "amountIn", "type": "uint256"}, {"name": "path", "type": "address[]"}],
        "outputs": [{"name": "amounts", "type": "uint256[]"}],
    },
    {
        "name": "swapExactETHForTokens",
        "type": "function",
        "stateMutability": "payable",
        "inputs": [
            {"name": "amountOutMin", "type": "uint256"},
            {"name": "path", "type": "address[]"},
            {"name": "to", "type": "address"},
            {"name": "deadline", "type": "uint256"},
        ],
        "outputs": [{"name": "amounts", "type": "uint256[]"}],
    },
    {
        "name": "swapExactTokensForETH",
        "type": "function",
        "stateMutability": "nonpayable",
        "inputs": [
            {"name": "amountIn", "type": "uint256"},
            {"name": "amountOutMin", "type": "uint256"},
            {"name": "path", "type": "address[]"},
            {"name": "to", "type": "address"},
            {"name": "deadline", "type": "uint256"},
        ],
        "outputs": [{"name": "amounts", "type": "uint256[]"}],
    },
    {
        "name": "swapExactTokensForTokens",
        "type": "function",
        "stateMutability": "nonpayable",
        "inputs": [
            {"name": "amountIn", "type": "uint256"},
            {"name": "amountOutMin", "type": "uint256"},
            {"name": "path", "type": "address[]"},
            {"name": "to", "type": "address"},
            {"name": "deadline", "type": "uint256"},
        ],
        "outputs": [{"name": "amounts", "type": "uint256[]"}],
    },
    # Supporting fee-on-transfer variants (we'll parse inputs and mirror with standard versions)
    {
        "name": "swapExactETHForTokensSupportingFeeOnTransferTokens",
        "type": "function",
        "stateMutability": "payable",
        "inputs": [
            {"name": "amountOutMin", "type": "uint256"},
            {"name": "path", "type": "address[]"},
            {"name": "to", "type": "address"},
            {"name": "deadline", "type": "uint256"},
        ],
        "outputs": [],
    },
    {
        "name": "swapExactTokensForETHSupportingFeeOnTransferTokens",
        "type": "function",
        "stateMutability": "nonpayable",
        "inputs": [
            {"name": "amountIn", "type": "uint256"},
            {"name": "amountOutMin", "type": "uint256"},
            {"name": "path", "type": "address[]"},
            {"name": "to", "type": "address"},
            {"name": "deadline", "type": "uint256"},
        ],
        "outputs": [],
    },
    {
        "name": "swapExactTokensForTokensSupportingFeeOnTransferTokens",
        "type": "function",
        "stateMutability": "nonpayable",
        "inputs": [
            {"name": "amountIn", "type": "uint256"},
            {"name": "amountOutMin", "type": "uint256"},
            {"name": "path", "type": "address[]"},
            {"name": "to", "type": "address"},
            {"name": "deadline", "type": "uint256"},
        ],
        "outputs": [],
    },
]


# ----------------------------
# Configuration
# ----------------------------

@dataclass
class Config:
    rpc_url: str
    target_wallet: str
    private_key: str
    my_address: str
    slippage_bps: int
    poll_seconds: int
    simulation_mode: bool
    amount_multiplier_bp: Optional[int]
    fixed_input_bnb: Optional[Decimal]
    chain_id: int = 56  # BSC mainnet
    router_address: str = ROUTER_V2_ADDRESS
    wbnb_address: str = WBNB_ADDRESS
    gas_price_gwei_ceiling: Optional[int] = None  # optional ceiling; None = use network gas price


def get_env_config() -> Config:
    rpc_url = os.getenv("BSC_RPC_URL", "https://bsc-dataseed.binance.org/")
    target_wallet = os.getenv("TARGET_WALLET", "").strip()
    priv = os.getenv("PRIVATE_KEY", "").strip()
    if not target_wallet or not priv:
        print("ERROR: TARGET_WALLET and PRIVATE_KEY must be set as environment variables.", file=sys.stderr)
        sys.exit(2)

    # Derive my address from private key (later via web3)
    # Placeholder for now; filled after web3 init
    my_address = ""

    slippage_bps = int(os.getenv("SLIPPAGE_BPS", "100"))  # default 1%
    poll_seconds = int(os.getenv("POLL_SECONDS", "7"))
    simulation_mode = os.getenv("SIMULATION_MODE", "1") not in ("0", "false", "False")
    amount_multiplier_bp = os.getenv("AMOUNT_MULTIPLIER_BP")
    fixed_input_bnb = os.getenv("FIXED_INPUT_BNB")

    amt_mult = int(amount_multiplier_bp) if amount_multiplier_bp else None
    fixed_bnb = Decimal(fixed_input_bnb) if fixed_input_bnb else None

    gas_price_gwei_ceiling = os.getenv("GAS_PRICE_GWEI_CEILING")
    gas_ceiling = int(gas_price_gwei_ceiling) if gas_price_gwei_ceiling else None

    return Config(
        rpc_url=rpc_url,
        target_wallet=AsyncWeb3.to_checksum_address(target_wallet),
        private_key=priv,
        my_address=my_address,  # set after web3 is created
        slippage_bps=slippage_bps,
        poll_seconds=poll_seconds,
        simulation_mode=simulation_mode,
        amount_multiplier_bp=amt_mult,
        fixed_input_bnb=fixed_bnb,
        gas_price_gwei_ceiling=gas_ceiling,
    )


# ----------------------------
# Logging
# ----------------------------

def setup_logging() -> None:
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler("copy_trader.log")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)


# ----------------------------
# Web3 Connection & Contracts
# ----------------------------

@dataclass
class Web3Context:
    w3: AsyncWeb3
    router: AsyncContract

async def connect_web3(cfg: Config) -> Web3Context:
    w3 = AsyncWeb3(AsyncHTTPProvider(cfg.rpc_url), modules={"eth": (AsyncWeb3.eth,)})
    is_connected = await w3.is_connected()
    if not is_connected:
        logging.error("Failed to connect to RPC at %s", cfg.rpc_url)
        sys.exit(2)

    # Derive my address from private key
    acct = w3.eth.account.from_key(cfg.private_key)
    cfg.my_address = acct.address
    logging.info("Connected to BSC RPC. My address: %s | Target: %s | Simulation: %s",
                 cfg.my_address, cfg.target_wallet, cfg.simulation_mode)

    router = await w3.eth.contract(address=AsyncWeb3.to_checksum_address(cfg.router_address),
                                   abi=ROUTER_V2_MIN_ABI)
    return Web3Context(w3=w3, router=router)


# ----------------------------
# Utils (ERC-20 helpers, math)
# ----------------------------

async def erc20(w3: AsyncWeb3, token: str) -> AsyncContract:
    return await w3.eth.contract(address=AsyncWeb3.to_checksum_address(token), abi=ERC20_MIN_ABI)

async def token_decimals(w3: AsyncWeb3, token: str) -> int:
    if token.lower() == WBNB_ADDRESS.lower():
        return 18
    c = await erc20(w3, token)
    return await c.functions.decimals().call()

async def token_symbol(w3: AsyncWeb3, token: str) -> str:
    if token.lower() == WBNB_ADDRESS.lower():
        return "WBNB"
    c = await erc20(w3, token)
    try:
        return await c.functions.symbol().call()
    except Exception:
        return token[:6]  # fallback

async def token_balance(w3: AsyncWeb3, token: str, owner: str) -> int:
    if token.lower() == WBNB_ADDRESS.lower():
        # Native BNB balance (not WBNB), but many paths use WBNB token.
        return await w3.eth.get_balance(owner)
    c = await erc20(w3, token)
    return await c.functions.balanceOf(owner).call()

async def ensure_allowance(w3: AsyncWeb3, token: str, owner: str, spender: str, required: int,
                           cfg: Config) -> Optional[str]:
    """
    Ensure router has sufficient allowance for `token`. If not, submit an approve tx.
    Returns tx hash if approval was sent; None otherwise.
    """
    if token.lower() == WBNB_ADDRESS.lower():
        # When path starts with WBNB, the input is native BNB for swapExactETH*; no approve.
        return None

    c = await erc20(w3, token)
    current = await c.functions.allowance(owner, spender).call()
    if current >= required:
        return None

    # Approve max uint256 to minimize future approvals
    max_uint = 2**256 - 1

    nonce = await w3.eth.get_transaction_count(owner)
    gas_price = await w3.eth.gas_price
    if cfg.gas_price_gwei_ceiling:
        gas_price = min(gas_price, AsyncWeb3.to_wei(cfg.gas_price_gwei_ceiling, "gwei"))

    tx = await c.functions.approve(spender, max_uint).build_transaction({
        "from": owner,
        "nonce": nonce,
        "gasPrice": gas_price,
        "chainId": cfg.chain_id,
    })
    gas_est = await w3.eth.estimate_gas(tx)
    tx["gas"] = math.ceil(gas_est * 1.2)

    if cfg.simulation_mode:
        logging.info("[SIM] Approve %s for router (max). Gas est=%s", await token_symbol(w3, token), tx["gas"])
        return None

    signed = w3.eth.account.sign_transaction(tx, private_key=cfg.private_key)
    tx_hash = await w3.eth.send_raw_transaction(signed.rawTransaction)
    logging.info("Sent approval tx: %s", tx_hash.hex())
    # Wait for receipt
    rec = await w3.eth.wait_for_transaction_receipt(tx_hash)
    if rec.status != 1:
        logging.error("Approval tx failed: %s", tx_hash.hex())
        raise RuntimeError("Approval failed")
    return tx_hash.hex()

def bps(value: int, bps_val: int) -> int:
    return (value * bps_val) // 10_000

def apply_slippage_min_out(quoted_out: int, slippage_bps: int) -> int:
    return quoted_out - bps(quoted_out, slippage_bps)

def now_deadline(seconds: int = 300) -> int:
    return int(time.time()) + seconds


# ----------------------------
# Decoding target swaps
# ----------------------------

SWAP_FN_NAMES = {
    "swapExactETHForTokens",
    "swapExactTokensForETH",
    "swapExactTokensForTokens",
    "swapExactETHForTokensSupportingFeeOnTransferTokens",
    "swapExactTokensForETHSupportingFeeOnTransferTokens",
    "swapExactTokensForTokensSupportingFeeOnTransferTokens",
}

async def decode_router_call(ctx: Web3Context, tx: TxData) -> Optional[Tuple[str, Dict[str, Any]]]:
    """
    Return (function_name, args_dict) if tx is a supported router swap; else None.
    """
    if not tx["to"]:
        return None
    if AsyncWeb3.to_checksum_address(tx["to"]) != AsyncWeb3.to_checksum_address(ROUTER_V2_ADDRESS):
        return None
    try:
        fn, args = ctx.router.decode_function_input(tx["input"])
        if fn.fn_name in SWAP_FN_NAMES:
            return fn.fn_name, args
    except Exception:
        return None
    return None


# ----------------------------
# Sizing logic
# ----------------------------

async def compute_our_amount_in(
    w3: AsyncWeb3,
    cfg: Config,
    fn_name: str,
    args: Dict[str, Any],
    my_addr: str,
) -> Tuple[int, List[str], bool]:
    """
    Returns (our_amount_in_wei, path, is_native_in)
    - For ETH/BNB-in functions, amountIn is tx.value (we don't see it in args). We'll size via FIXED_INPUT_BNB or multiplier vs target.
    - For token-in functions, size via multiplier of target amountIn, capped by our balance.
    """
    path: List[str] = [AsyncWeb3.to_checksum_address(p) for p in args["path"]]
    is_native_in = (fn_name.startswith("swapExactETHFor"))

    if is_native_in:
        # Our BNB balance:
        my_bnb = await w3.eth.get_balance(my_addr)
        amt_from_fixed = None
        if cfg.fixed_input_bnb is not None:
            amt_from_fixed = int(AsyncWeb3.to_wei(cfg.fixed_input_bnb, "ether"))

        if amt_from_fixed is not None:
            our_amt_in = min(amt_from_fixed, my_bnb)
        else:
            # multiplier against target's value is tricky here because we don't have target's value directly
            # (we could fetch from tx.value once we load tx fully, caller should pass tx.value if needed)
            # For simplicity: use multiplier against our balance (conservative).
            mult_bp = cfg.amount_multiplier_bp or 0
            our_amt_in = bps(my_bnb, mult_bp)
        # Leave some dust for gas
        reserve_gas = AsyncWeb3.to_wei(0.003, "ether")  # ~0.003 BNB
        our_amt_in = max(0, our_amt_in - reserve_gas)
        return our_amt_in, path, True

    else:
        # Token-in functions: args contains amountIn
        target_amount_in = int(args["amountIn"])
        mult_bp = cfg.amount_multiplier_bp or 0
        our_desired = (target_amount_in * mult_bp) // 10_000
        src_token = path[0]
        my_bal = await token_balance(w3, src_token, my_addr)
        our_amt_in = min(our_desired, my_bal)
        return our_amt_in, path, False


# ----------------------------
# Trade execution
# ----------------------------

async def quote_out(ctx: Web3Context, amount_in: int, path: List[str]) -> List[int]:
    return await ctx.router.functions.getAmountsOut(amount_in, path).call()

async def build_and_send_swap(
    ctx: Web3Context,
    cfg: Config,
    fn_name: str,
    path: List[str],
    our_amount_in: int,
    is_native_in: bool,
) -> Optional[str]:
    """
    Builds and (optionally) sends the swap tx for our wallet.
    Returns tx hash hex if sent, None if simulated or skipped.
    """
    if our_amount_in <= 0:
        logging.warning("Our amount_in is 0; skipping.")
        return None

    my_addr = cfg.my_address
    deadline = now_deadline(300)

    # Quote expected amounts for slippage calculation
    try:
        amounts = await quote_out(ctx, our_amount_in, path)
        quoted_out = int(amounts[-1])
        min_out = apply_slippage_min_out(quoted_out, cfg.slippage_bps)
    except Exception as e:
        logging.error("Failed to quote getAmountsOut: %s", e)
        return None

    # Prepare tx params
    nonce = await ctx.w3.eth.get_transaction_count(my_addr)
    gas_price = await ctx.w3.eth.gas_price
    if cfg.gas_price_gwei_ceiling:
        gas_price = min(gas_price, AsyncWeb3.to_wei(cfg.gas_price_gwei_ceiling, "gwei"))

    common = {
        "from": my_addr,
        "nonce": nonce,
        "gasPrice": gas_price,
        "chainId": cfg.chain_id,
    }

    # Approvals if input is ERC-20
    if not is_native_in:
        src_token = path[0]
        try:
            await ensure_allowance(ctx.w3, src_token, my_addr, cfg.router_address, our_amount_in, cfg)
        except Exception as e:
            logging.error("Approval failed or not confirmed: %s", e)
            return None

    # Choose appropriate router method to mirror swap direction
    tx = None
    if is_native_in:
        # Use swapExactETHForTokens
        tx = await ctx.router.functions.swapExactETHForTokens(
            min_out, path, my_addr, deadline
        ).build_transaction({
            **common,
            "value": our_amount_in,
        })
    else:
        # Determine if path ends with WBNB => tokens->ETH (BNB); otherwise tokens->tokens
        if path[-1].lower() == WBNB_ADDRESS.lower():
            # tokens -> BNB
            tx = await ctx.router.functions.swapExactTokensForETH(
                our_amount_in, min_out, path, my_addr, deadline
            ).build_transaction(common)
        else:
            # tokens -> tokens
            tx = await ctx.router.functions.swapExactTokensForTokens(
                our_amount_in, min_out, path, my_addr, deadline
            ).build_transaction(common)

    # Estimate gas and pad
    try:
        gas_est = await ctx.w3.eth.estimate_gas(tx)
        tx["gas"] = math.ceil(gas_est * 1.2)
    except Exception as e:
        logging.error("Gas estimation failed: %s", e)
        return None

    # Pretty log
    try:
        src_sym = await token_symbol(ctx.w3, path[0])
        dst_sym = await token_symbol(ctx.w3, path[-1])
        src_dec = await token_decimals(ctx.w3, path[0])
        dst_dec = await token_decimals(ctx.w3, path[-1])
        human_in = Decimal(our_amount_in) / (Decimal(10) ** src_dec)
        human_out_min = Decimal(min_out) / (Decimal(10) ** dst_dec)
        logging.info(
            "Executing swap %s -> %s | amount_in=%s %s | min_out=%s %s | gas≈%s | slippage=%sbps",
            src_sym, dst_sym, human_in, src_sym, human_out_min, dst_sym, tx.get("gas"),
            cfg.slippage_bps
        )
    except Exception:
        pass

    if cfg.simulation_mode:
        logging.info("[SIM] Would send swap tx (not broadcasting).")
        return None

    # Sign & send
    try:
        signed = ctx.w3.eth.account.sign_transaction(tx, private_key=cfg.private_key)
        tx_hash = await ctx.w3.eth.send_raw_transaction(signed.rawTransaction)
        logging.info("Broadcasted swap tx: %s", tx_hash.hex())
        rec = await ctx.w3.eth.wait_for_transaction_receipt(tx_hash)
        if rec.status != 1:
            logging.error("Swap tx failed: %s", tx_hash.hex())
            return None
        logging.info("Swap confirmed in block %s", rec.blockNumber)
        return tx_hash.hex()
    except Exception as e:
        logging.error("Broadcast failed: %s", e)
        return None


# ----------------------------
# Block polling & event loop
# ----------------------------

async def fetch_latest_block_number(w3: AsyncWeb3) -> int:
    return await w3.eth.block_number

async def get_block_txs(w3: AsyncWeb3, block_number: int) -> List[TxData]:
    block = await w3.eth.get_block(block_number, full_transactions=True)
    return block["transactions"] or []

async def process_block(
    ctx: Web3Context,
    cfg: Config,
    block_number: int,
) -> None:
    txs = await get_block_txs(ctx.w3, block_number)
    if not txs:
        return

    for tx in txs:
        # Filter: from target wallet & to = router
        if (tx.get("from") or "").lower() != cfg.target_wallet.lower():
            continue
        if not tx.get("to"):
            continue
        if AsyncWeb3.to_checksum_address(tx["to"]) != AsyncWeb3.to_checksum_address(cfg.router_address):
            continue

        # Confirm tx success (avoid pending/reorg issues)
        try:
            receipt: TxReceipt = await ctx.w3.eth.get_transaction_receipt(tx["hash"])
            if receipt.status != 1:
                continue
        except Exception:
            continue

        decoded = await decode_router_call(ctx, tx)
        if not decoded:
            continue

        fn_name, args = decoded
        logging.info("Detected target swap: %s | hash=%s", fn_name, tx["hash"].hex())

        # Decide our input amount & path
        try:
            our_amount_in, path, is_native_in = await compute_our_amount_in(
                ctx.w3, cfg, fn_name, args, cfg.my_address
            )

            # Special handling: if fn is ETH-in, we may want to use tx.value as a guide for multiplier mode.
            # If multiplier mode and FIXED_INPUT_BNB isn't set, scale by our multiplier of target value.
            if fn_name.startswith("swapExactETHFor"):
                if cfg.fixed_input_bnb is None and cfg.amount_multiplier_bp:
                    target_value = int(tx.get("value", 0))
                    # reserve gas
                    reserve_gas = AsyncWeb3.to_wei(0.003, "ether")
                    scaled = bps(target_value, cfg.amount_multiplier_bp)
                    my_bal = await ctx.w3.eth.get_balance(cfg.my_address)
                    our_amount_in = max(0, min(scaled, my_bal - reserve_gas))

            # If we don't hold required source token (non-native) => skip
            if not is_native_in:
                src_bal = await token_balance(ctx.w3, path[0], cfg.my_address)
                if src_bal < our_amount_in or our_amount_in == 0:
                    sym = await token_symbol(ctx.w3, path[0])
                    logging.warning("Insufficient %s balance (%s). Skipping mirror.", sym, src_bal)
                    continue

            # Execute
            await build_and_send_swap(ctx, cfg, fn_name, path, our_amount_in, is_native_in)
        except Exception as e:
            logging.exception("Failed to mirror trade: %s", e)


async def poll_loop(ctx: Web3Context, cfg: Config, stop_event: asyncio.Event) -> None:
    last_block = await fetch_latest_block_number(ctx.w3)
    logging.info("Starting from latest block: %s", last_block)

    while not stop_event.is_set():
        try:
            latest = await fetch_latest_block_number(ctx.w3)
            if latest > last_block:
                for b in range(last_block + 1, latest + 1):
                    logging.debug("Processing block %s", b)
                    await process_block(ctx, cfg, b)
                last_block = latest
        except Exception as e:
            logging.error("Polling error: %s", e)
        await asyncio.wait([stop_event.wait()], timeout=cfg.poll_seconds)


# ----------------------------
# Graceful shutdown
# ----------------------------

def setup_signal_handlers(loop: asyncio.AbstractEventLoop, stop_event: asyncio.Event) -> None:
    def _signal(name: str):
        logging.info("Received %s, shutting down...", name)
        stop_event.set()

    try:
        loop.add_signal_handler(signal.SIGINT, _signal, "SIGINT")
        loop.add_signal_handler(signal.SIGTERM, _signal, "SIGTERM")
    except NotImplementedError:
        # Windows
        signal.signal(signal.SIGINT, lambda *_: _signal("SIGINT"))
        signal.signal(signal.SIGTERM, lambda *_: _signal("SIGTERM"))


# ----------------------------
# Main
# ----------------------------

async def main_async() -> None:
    setup_logging()
    cfg = get_env_config()

    # Connect web3 and contracts
    ctx = await connect_web3(cfg)

    # Basic sanity logs
    bnb_bal = await ctx.w3.eth.get_balance(cfg.my_address)
    logging.info("BNB balance: %s BNB", AsyncWeb3.from_wei(bnb_bal, "ether"))

    # Start polling
    stop_event = asyncio.Event()
    setup_signal_handlers(asyncio.get_running_loop(), stop_event)
    logging.info("Entering polling loop (interval=%ss)...", cfg.poll_seconds)
    await poll_loop(ctx, cfg, stop_event)

def main() -> None:
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        print("\nInterrupted by user.")

if __name__ == "__main__":
    main()
