# Painting Environment First Contact

限られた色と単純な筆を使い、目標画像へ逐次的に近づくための
最小の Reinforcement Learning Environment です。

このリポジトリの目的は、写実的な絵具物理を再現することではありません。

> 「描く」という制作行為を、Observation / Action / Reward を持つ
> 学習可能なEnvironmentとして定義する

ことが目的です。

## 最小モデル

- Canvas: 64 × 64 RGB
- Palette: 5色
- Brush: 不透明な円形スタンプ
- Action: `(x, y, color_index, radius)`
- Observation: 現在のCanvasとTarget
- Reward: Action前後の画像誤差の改善量
- Episode終了:
  - 最大ステップ数に到達
  - 十分にTargetへ近づいた

```text
Current Canvas
      ↓
Select Paint Action
      ↓
Apply Circular Brush
      ↓
Observe New Canvas
      ↓
Reward = Error Before - Error After
```
## Python on Ubuntu

```bash
sudo apt update
sudo apt install -y \
    python3 \
    python3-venv \
    python3-pip \
    python-is-python3
```

## セットアップ

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e .
```

## 実行

Random Agent:

```bash
python examples/run_random.py
```

Greedy Agent:

```bash
python examples/run_greedy.py
```

出力は `outputs/` に保存されます。

## テスト

```bash
pytest
```

## なぜ最初からRLを学習しないのか

最初に確認すべきなのは、ニューラルネットワークではなくEnvironmentです。

1. ActionでCanvasが変化する
2. 良いActionでRewardが正になる
3. Baseline AgentがRandom Agentより改善する
4. 問題設定が学習可能な形になっている

この4点を確認してから、PPO、SAC、DQNなどの学習器を接続します。

なお、このEnvironmentのActionは複数の離散・連続要素を含むため、
最初の学習器としてはActionを離散化した版、または独自Policyを用意するのが自然です。

## 発展ロードマップ

### v0: Stamp
- 円形スタンプ
- 固定Palette
- 上書きのみ

### v1: Alpha
- Paint量
- 半透明
- 重ね塗り

### v2: Mixing
- 限られた顔料からの混色
- RGB直接指定を禁止

### v3: Stroke
- 始点、終点、幅を持つStroke
- Flat Brush / Round Brush

### v4: Perception
- 観察距離を考慮したReward
- Blur、Multi-scale、Feature similarity

### v5: Sim-to-Real
- 実寸CanvasとBrushの比率
- Camera Feedback
- Robot Actionへの写像

## 構造

```text
painting-environment-first-contact/
├── painting_env/
│   ├── __init__.py
│   ├── env.py
│   ├── targets.py
│   └── agents.py
├── examples/
│   ├── run_random.py
│   └── run_greedy.py
├── tests/
│   └── test_env.py
├── pyproject.toml
└── README.md
```

## 中心となる問い

この最小版が扱う問いは、次の一つです。

> 限られた大きさの筆と限られた色を使い、
> どの順序でPaint Actionを置けば、
> Targetに効率よく近づけるか。
