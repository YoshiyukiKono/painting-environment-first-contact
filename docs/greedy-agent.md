# Greedy Agent

## 1. Greedyとは何か

Greedy Agentは、現在の状態から見て、
直後の画像誤差を最も減らすActionを選ぶ。

未来の複数ステップは考慮しない。
学習も行わない。

## 2. この実装の処理

1. 現在のCanvasを保存する
2. 現在のTarget誤差を計算する
3. 候補ActionをランダムにN個生成する
4. 各候補を仮にCanvasへ適用する
5. 適用後の誤差を測る
6. 最も誤差を減らしたActionを採用する
7. これをEpisode終了まで繰り返す

## 3. 擬似コード

for each step:
    best_action = None
    best_error = current_error

    for candidate in random_actions:
        simulated_canvas = apply(candidate)
        error = compare(simulated_canvas, target)

        if error < best_error:
            best_action = candidate
            best_error = error

    execute(best_action)

## 4. なぜこれだけで描けるのか

- Rewardと画像誤差が直接接続されている
- 各Actionの効果が局所的で予測しやすい
- StampがTargetの局所領域を塗り替える
- 大きな構造から小さな修正へ自然に進む
- 多数の候補から選択することで、
  単純なRandom Agentより効率的になる

## 5. Greedy AgentはRLなのか

Greedy Agent自体はRLではない。

Policyを学習せず、
各ステップでEnvironmentを直接試行している。

役割は、RLを導入する前のBaselineである。

## 6. このBaselineが証明すること

Greedy AgentがRandom Agentより良ければ、

- ActionがCanvasへ正しく反映されている
- Rewardが有意味である
- Targetへ近づくActionが存在する
- 問題設定に最低限の学習可能性がある

ことを確認できる。

## 7. Greedy Agentの限界

- 未来の塗り重ねを考えない
- 一時的に悪化するActionを選べない
- 候補に含まれない良いActionを発見できない
- 候補数を増やすほど計算量が増える
- 混色や半透明では局所評価が難しくなる
- 実物の絵具ではActionを巻き戻して試せない

## 8. RLが必要になる境界

以下を導入するとGreedyだけでは難しくなる。

- 半透明
- 混色
- Stroke
- 絵具量
- 筆圧
- 乾燥
- 描画順序
- 一度塗ると戻せない物理制約
- 複数ステップ後の見た目を重視するReward


Greedy Agentが成功することは、RLが不要であることを意味しない。
それは、現在のEnvironmentが「局所改善で解ける最小問題」として正しく設計されていることを示している。
