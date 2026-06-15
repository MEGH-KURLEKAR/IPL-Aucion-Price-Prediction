# 🏏 IPL Auction Price Prediction — Pressure Index & Player Valuation

> Predicting IPL auction prices for batsmen and bowlers using engineered pressure metrics and ball-by-ball match data.

---

## 📌 Project Overview

This project builds a machine learning pipeline to predict **IPL player auction prices** using performance statistics derived from raw ball-by-ball delivery data. The key innovation is a custom **Pressure Index** — a composite metric engineered from required run rate, wickets lost, and match context — used as a feature in Random Forest and XGBoost regression models.

---

## 📂 Dataset

- **`ultimate.csv`** — Ball-by-ball IPL delivery data (~79,000 rows) with columns:
  - `match_id`, `season`, `venue`, `innings`, `ball`
  - `batting_team`, `bowling_team`, `striker`, `non_striker`, `bowler`
  - `runs_off_bat`, `extras`, `wides`, `noballs`, `wicket_type`, `player_dismissed`
- **Batter/Bowler summary files** — Aggregated player stats merged with auction price data

---

## ⚙️ Feature Engineering

Key engineered features built from raw delivery data:

| Feature | Description |
|---|---|
| `runs_till_now` | Cumulative runs per innings (ball-by-ball) |
| `run_rate` | Runs scored / cumulative overs at each delivery |
| `required_runs` | Runs needed by chasing team at each ball |
| `required_run_rate` | RRR at each delivery in the second innings |
| `remaining_balls` | Balls left in the innings |
| `wickets_lost` | Cumulative wickets fallen per innings |
| `batsman_pressure_index` | Custom composite pressure score for batters |
| `bowler_pressure_index` | Custom composite pressure score for bowlers |

### Pressure Index Formula (Batting)

```
PI = (RRR / initial_RRR × 100) + (wickets × 18 / 180 × target) × (remaining_balls / total_balls) × (remaining_runs / target)
```

Handles regular innings (120 balls) and super overs (6 balls) separately.

---

## 🤖 Models

### Batting Price Prediction
- Features: ball-by-ball batting stats + `batsman_pressure_index` + team (one-hot encoded)
- Models: **Random Forest**, **XGBoost**

### Bowling Price Prediction
- Features: aggregated bowling stats (`economy`, `strike_rate`, `dot_ball_%`, `boundary_concession_%`, `bowler_pressure_index`) + team
- Models: **Random Forest**, **XGBoost**

---

## 📊 Results

| Model | Target | R² Score |
|---|---|---|
| Random Forest | Batter Auction Price | 0.82 |
| XGBoost | Batter Auction Price | In progress |
| Random Forest | Bowler Auction Price | See notebook |
| XGBoost | Bowler Auction Price | See notebook |

> ✦ **82% R²** on batter auction price prediction — 20% improvement over baseline

---

## 🛠️ Tech Stack

`Python` · `Pandas` · `Scikit-learn` · `XGBoost` · `Matplotlib` · `Seaborn` · `Google Colab`

---

## 📁 Repository Structure

```
ipl-auction-price-prediction/
│
├── crick_attempt2_4_.ipynb   # Main notebook (EDA → Feature Eng → Modeling)
├── ultimate.csv              # Ball-by-ball delivery data
└── README.md
```

---

## 🚀 How to Run

1. Clone the repo and open `crick_attempt2_4_.ipynb` in Google Colab or Jupyter
2. Upload `ultimate.csv` to the same directory (or your Colab session)
3. Run all cells top to bottom

---

## 💡 Key Takeaways

- Ball-by-ball data enables rich pressure context that aggregate stats miss
- The custom Pressure Index captures clutch performance under high-stakes match situations
- Super over handling required separate logic (6-ball vs 120-ball innings)
- Run rate momentum and wicket context are stronger predictors of auction value than raw counting stats

---
