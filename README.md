# copy_trader.py

Real-time copy trader for PancakeSwap Router V2 on BSC (Python 3.10+).

## What it does
- Monitors a target wallet’s swaps on **PancakeSwap V2 Router** and replicates them on your wallet.
- Async polling (5–10s) using **AsyncWeb3**.
- Slippage protection via `getAmountsOut`.
- Optional **proportional sizing** or **fixed input size**.
- Handles approvals, gas estimation, retries, structured logging, and graceful shutdown.

> ⚠️ **Use at your own risk.** Test in **simulation mode** first and never commit private keys.

---

## Quick Start

### Install
```bash
pip install "web3==6.*" "eth_abi==5.*" "python-dotenv==1.*"
