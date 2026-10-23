# Almanacminer

Custom forecasting miners for **Almanac / Bittensor Subnet 41 (IM2)**.

This repo contains:

- `phase1_starter.py` — a low-cost search + forecast baseline.
- `phase2_research.py` — a stronger market-anchored research miner.
- `test_agents.py` — offline smoke tests that use fake provider responses and spend no portal credits.

## Current Almanac scoring target

The current upstream scoring code uses:

- **70% Accuracy** — Brier skill relative to the market price at prediction time.
- **30% Edge** — market-relative Brier difference + PnL signal.

Important current gates include:

- at least **10 valid resolved predictions**;
- invalid rate below **15%**;
- recency-weighted mean Brier below **0.25**;
- inactivity fade after **24 hours** and hard zero after **72 hours**;
- recency half-life of **14 days**.

Phase 2 is built around that model. It starts from the live YES price when available, researches both the current case and the strongest counter-case, estimates an independent probability, then shrinks that estimate back toward the market unless the evidence is strong.

## Recommended layout

Clone this repo beside the official Almanac repo:

```text
workspace/
├── almanac/
└── Almanacminer/
```

```bash
git clone https://github.com/corvxai/almanac.git
git clone https://github.com/miczino/Almanacminer.git
cd almanac
```

## Install the official miner environment

Python 3.10+ is required.

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The upstream repo pins Bittensor and installs `btcli`. Do not separately install the legacy `bittensor-cli` package.

## Configure your Almanac Portal key

Create `almanac/.env`:

```env
GATEWAY_API_KEY=your-gateway-api-key
```

Never commit that file.

Check your portal balance:

```bash
python miner/cli.py balance
```

## Run offline smoke tests first

From the `Almanacminer` repo:

```bash
python test_agents.py
```

If the official repo is elsewhere:

### Windows PowerShell

```powershell
$env:ALMANAC_REPO="C:\path\to\almanac"
python test_agents.py
```

### Linux / macOS

```bash
ALMANAC_REPO=/path/to/almanac python test_agents.py
```

Expected output:

```text
All Almanac miner smoke tests passed.
```

## Phase 1 live test

From the official `almanac` repo:

```bash
python miner/cli.py test-agent ../Almanacminer/phase1_starter.py
```

Phase 1 normally makes two provider calls:

1. current online research;
2. final probability synthesis.

Do not submit until the CLI prints:

```text
Agent result validated successfully.
```

## Phase 2 live test

```bash
python miner/cli.py test-agent ../Almanacminer/phase2_research.py
```

Phase 2 normally makes three provider calls on production assignments:

1. current-state research;
2. adversarial/counter-case research;
3. independent probability synthesis.

If a local `test-agent` event does not expose the market price, Phase 2 performs one extra online market lookup.

### Phase 2 flow

```text
Live YES market prior
        ↓
Current-state research
        ↓
Adversarial research
        ↓
Independent estimate
        ↓
Evidence-strength estimate
        ↓
Conservative shrinkage toward market
        ↓
Final P(YES)
```

## Create/register your Bittensor miner

Example:

```bash
btcli wallet create --wallet almanac --wallet-hotkey miner
```

Testnet rehearsal:

```bash
btcli subnets register \
  --netuid 172 \
  --wallet almanac \
  --wallet-hotkey miner \
  --network test
```

Mainnet SN41:

```bash
btcli subnets register \
  --netuid 41 \
  --wallet almanac \
  --wallet-hotkey miner \
  --network finney
```

Review the TAO registration cost before confirming.

## Submit Phase 1

From the official repo:

```bash
python miner/cli.py submit-agent \
  ../Almanacminer/phase1_starter.py \
  --wallet-name almanac \
  --wallet-hotkey-name miner
```

## Upgrade to Phase 2

After Phase 2 passes `test-agent`:

```bash
python miner/cli.py submit-agent \
  ../Almanacminer/phase2_research.py \
  --wallet-name almanac \
  --wallet-hotkey-name miner
```

A successful new submission replaces the current active agent.

## Security

Never commit:

- `.env`
- Almanac gateway API keys
- Bittensor seed phrases
- coldkey/private-key files
- wallet backups

The agents contain no embedded secrets. All external calls go through Almanac's `ctx.call_provider(...)`.

## Upstream

Official Almanac repo:

https://github.com/corvxai/almanac
