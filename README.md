# copy_trader.py

Real-time copy trader for PancakeSwap Router V2 on BSC (Python 3.10+).

## What it does
- Monitors a target wallet’s swaps on **PancakeSwap V2 Router** and replicates them on your wallet.
- Async polling (5–10s) using **AsyncWeb3**.
- Slippage protection via `getAmountsOut`.
- Optional **proportional sizing** or **fixed input size**.
- Handles approvals, gas estimation, retries, structured logging, and graceful shutdown.

⚠️ Big warning: Copy trading smart contracts is risky. This script is provided for educational use—use at your own risk. Always test with SIMULATION_MODE=1 first and a throwaway wallet on a small private RPC/fork before touching real funds.
> ⚠️ **Use at your own risk.** Test in **simulation mode** first and never commit private keys.

---

## Quick Start

### Install
```bash
pip install "web3==6.*" "eth_abi==5.*" "python-dotenv==1.*"

## Notes & tips

Simulation mode (SIMULATION_MODE=1) logs trades, quotes, gas, and what it would do, without sending transactions or approvals.

## Sizing:

If the target spends BNB (native) and you set FIXED_INPUT_BNB, the script uses that fixed amount (minus a small gas reserve).

Otherwise, it uses AMOUNT_MULTIPLIER_BP (e.g., 1000 = 10%) against the target’s input for ETH-in calls (using tx.value) or against the target’s amountIn for token-in calls, capped by your balance.

Approvals: If a mirrored swap requires an ERC-20 as input, the script auto-approves the router with max allowance (unless in simulation).

Gas: Uses network gasPrice (legacy) with a 20% gas limit buffer. You can set an optional ceiling via GAS_PRICE_GWEI_CEILING.

## Safety:

It only mirrors successful target transactions (checks receipt status==1).

It skips when you don’t hold the source token.

Uses getAmountsOut to compute a fresh minOut and then applies your SLIPPAGE_BPS.

## Extend:

Add support for other swap function variants (exact-output, multi-hop specifics) by extending ROUTER_V2_MIN_ABI and SWAP_FN_NAMES.

Add retry logic/backoff on transient RPC errors or add multiple RPC URLs (round-robin).

Persist last processed block across restarts (e.g., to a file) to avoid missing events.
