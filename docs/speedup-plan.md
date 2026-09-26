---
title: 高速化構想
---

# 高速化構想: Docker・高速化ライブラリ・CUDA グラフ・Kev-0.8B

作成: 2026-09-26。この文書は構想の段階で、まだ何も実装していない。

**2026-09-26 追記(実施中)**: 0.8B より先に、**Kev-4B とこれまでの学習結果を引き継いだまま** Docker と高速化ライブラリに移すことにした(CUDA グラフは無効のまま)。Kev のソースは、ビルドコンテキストに Windows の venv を含めないよう、デプロイ済みと同じコミット(`9098a8b`)を GitHub から取得する。実際の定義は `docker/Dockerfile` と `docker-compose.yml`。

## 1. 目的と、判断に使う指標

1世代にかかる時間を縮めて、同じ時間で回せる世代数を増やす。

| 指標 | 現状(Kev-4B、Windows ネイティブ) | 測り方 |
|---|---|---|
| 推論時間(1手) | 中央値 約360ms、90パーセンタイル 約610ms | `runs/replays/gen-XXX.json` の `latency_ms` |
| 学習時間(1手) | 約2.1〜2.2秒 | `runs/gen-XXX.train.log` の `s/rec` |
| 1世代の所要時間 | 約30〜35分(第5〜8世代) | `runs/generations.json` の `train.minutes` |
| VRAM | 配信用 約10GB、学習中 約13GB(16GB 中) | `nvidia-smi` |

遅い原因は3つある。

1. `flash-linear-attention` と `causal_conv1d` が入っておらず、Qwen3.5 の DeltaNet と畳み込みが遅い標準実装で動いている(推論と学習の両方)。
2. CUDA グラフを無効にしている(`KEV_CUDA_GRAPHS=0`、16GB での安定性を優先したため)。
3. 練習試合もテストも、1ゲームずつ・1手ずつ問い合わせていて、Kev サーバーのまとめ処理(バッチ)を使っていない。

## 2. 前提の確認結果(2026-09-26)

- Docker Desktop 28.0.4(WSL2 バックエンド、Linux エンジン)。ランタイムに `nvidia` がある
- GPU: RTX 5060 Ti 16GB、compute capability 12.0(sm_120)、ドライバ 591.86
- WSL2 のディストリビューション: `docker-desktop`(稼働中)、`Ubuntu-24.04`(停止中)

コンテナから GPU を使える見込みは高い。ただし、`--gpus all` で実際に `nvidia-smi` が通るかは、まだ試していない。

## 3. 全体の構成

```
Windows(ホスト)
├─ 配信サーバー  python -m kev_tetris.stream   … 8765(配信画面・操作ページ)。今のまま Windows で動かす
└─ D:\GitHub\kev-tetris\runs\  ←── bind mount で共有 ──┐
                                                        │
Docker コンテナ kev-rl(Linux、GPU)                     │
├─ 学習ループ  python -m kev_tetris.rl               … /work/runs に書く
├─ kev.serve(練習試合・テスト用)                      … コンテナ内のポート 8019
└─ kev.train                                          … Linux なので kev_train_win.py は不要
```

- 学習ループ・Kev の推論・学習は、すべてコンテナの中で完結させる。Kev の起動や一時停止(SIGSTOP/SIGCONT)も Linux 内で閉じる
- 配信サーバーは Windows に残す。やり取りは今と同じくファイルで行う(`rl_status.json`、`rl_control.json`、`rl_live.json`、`generations.json`、`replays/`)。
- 配信用の Kev(8011/8012)をどちらで動かすかは、段階5で決める。0.8B なら、コンテナ内でもう1本動かして、ポートを公開するのがよい
- Hugging Face のキャッシュはホストの `%USERPROFILE%\.cache\huggingface` をマウントし、モデルを再ダウンロードしない
- デプロイ済みの `D:\GitHub\kev`(Windows 用 venv)には手を入れない。コンテナ内では、同じソースを読み取り専用でマウントし、コンテナ専用の venv を作る

## 4. コンテナイメージ(案)

```dockerfile
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04     # nvcc 入り(causal-conv1d のビルド用)
ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 TORCH_CUDA_ARCH_LIST="12.0"
RUN apt-get update && apt-get install -y python3.12 python3.12-venv python3.12-dev git build-essential ninja-build \
    && rm -rf /var/lib/apt/lists/*
RUN python3.12 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
RUN pip install "torch==2.8.0" --index-url https://download.pytorch.org/whl/cu128
COPY kev/pyproject.toml kev/uv.lock /src/kev/                        # 依存だけ先に入れてキャッシュを効かせる
RUN pip install uv && cd /src/kev && uv pip install --system -r pyproject.toml --extra serve
RUN pip install flash-linear-attention                               # Triton ベース。Linux なら pip で入る見込み
RUN pip install --no-build-isolation causal-conv1d                  # sm_120 向けにソースからビルドする前提
RUN pip install numpy
WORKDIR /work
```

起動(docker compose の案):

```yaml
services:
  kev-rl:
    build: .
    gpus: all
    shm_size: "8gb"
    volumes:
      - D:/GitHub/kev:/src/kev:ro                  # Kev のソース(読み取り専用、実行時は editable install を /opt/venv に)
      - D:/GitHub/kev-tetris:/work                 # kev-tetris と runs/
      - ${USERPROFILE}/.cache/huggingface:/root/.cache/huggingface
    environment:
      KEV_HOME: /src/kev
      KEV_PYTHON: /opt/venv/bin/python
      KEV_CUDA_GRAPHS: "1"                          # 段階ごとに切り替える
    command: python -m kev_tetris.rl --generations 0
```

未確定の点:
- `causal-conv1d` と `flash-linear-attention` が sm_120 と torch 2.8 で動くか。動かなければ、ビルド時の `TORCH_CUDA_ARCH_LIST` を変えるか、該当ライブラリの新しい版を試す。
- `/src/kev` を読み取り専用にすると、Kev が作業フォルダに何かを書く場合に失敗する。そのときは書き込み可能なコピーを使う。

## 5. kev-tetris 側で必要な変更

| 変更 | 理由 |
|---|---|
| 操作ページでの「学習が生きているか」の判定を、プロセス番号から「状態ファイルが60秒以内に更新されたか」に変える | コンテナ内のプロセス番号は Windows から見えない。学習ループは一時停止中も約1秒ごとに状態を書いているので、判定の代わりになる |
| 操作ページの「学習開始」で、Windows の python ではなく `docker compose up -d kev-rl` を実行するよう切り替え可能にする | コンテナで学習を起動するため |
| 学習ループのプロセスが自分で `kev.train` を起動するとき、`kev_train_win.py` を通さない(Linux では不要) | `sys.platform` で分ける |
| `runs/` 内のパスを相対パスで記録する(一部は絶対パス `D:\...` になっている) | コンテナ内では `/work/runs` になるため |
| 練習試合とテストの複数ゲームを同時に進める(スレッドで並列に問い合わせる) | Kev サーバーのバッチ処理を活かす。これだけは Windows のままでも入れられる |

## 6. 進め方と、次に進む条件

実際の順番(2026-09-26 に変更): Docker 化を先に行い、複数ゲームの同時進行は後に回した。

| 段階 | 内容 | 次に進む条件 | 状態 |
|---|---|---|---|
| 1 | コンテナで `nvidia-smi` と torch の GPU 認識を確かめる | GPU が見えること | 完了(RTX 5060 Ti が見えた) |
| 2 | イメージを作る(Kev、torch 2.8 + CUDA 12.8、高速化ライブラリ) | ビルドが通り、ライブラリが読み込めること | 作業中 |
| 3 | コンテナで Kev-4B を CUDA グラフなしで動かし、Windows と推論・学習の速さを比べる | 1手の推論時間と `s/rec` が明らかに縮むこと | これから |
| 3b | 第10世代から、コンテナで学習を再開する | 世代が進むこと | これから |
| 3c | 複数ゲームの同時進行を入れる | 1世代の時間が縮み、成績が変わらないこと | 未着手 |
| 4 | Kev-0.8B + 高速化ライブラリ + CUDA グラフで速さと安定性を測る | 数時間動かしても落ちないこと |
| 5 | 4B の最良世代から 0.8B に蒸留する(下記) | 蒸留後の 0.8B のテスト成績が、元の 4B の世代に十分近いこと |
| 6 | 0.8B で強化学習を続ける。配信用と学習用の Kev を同時に動かし、次の世代の先読み込みも有効にする | — |

### 0.8B への蒸留(段階5)

以前、Kev-0.8B を学習なしでそのまま使ったときは、32手でゲームオーバー・0ラインだった。ゼロから強化学習をやり直すのは遅いので、4B の知識を移す。

1. 4B の最良世代(第6世代、または v2 系統の良い世代)に、ランダムなしで多数のゲーム(例: 100ゲーム)を打たせる。「盤面と、4B が選んだ手」の記録を作る
2. その記録を正解ラベルにして、Kev-0.8B(`jaredpalmer/kev-0.8b`)を `kev.train` で追加学習する。これは教師ありの学習で、強化学習ではない
3. 蒸留した 0.8B を第0世代として、同じテストで成績を比べる
4. 良ければ、これを出発点に 0.8B の系統で強化学習を続ける(世代の記録は、4B の系統と区別して残す)

## 7. 0.8B が駄目だった場合の代替

「0.8B が駄目」とは、次のどちらかの場合を指す。
- 蒸留しても、テストの成績が元の 4B に大きく届かない
- 強化学習を続けても、4B の系統より伸びが明らかに悪い

その場合は **Kev-4B + 高速化ライブラリ(CUDA グラフなし)** に戻す。

| 構成 | 速さ | VRAM | 安定性 | 位置づけ |
|---|---|---|---|---|
| 4B、ライブラリなし、グラフなし(現状) | 基準 | 学習中 約13GB | 実績あり | 現在の構成 |
| **4B、ライブラリあり、グラフなし** | 速くなる見込み | 同程度 | 安定する見込み | **0.8B が駄目なときの第一候補** |
| 4B、ライブラリあり、グラフあり | さらに速い可能性 | 余裕が減る | 16GB で不安定になるおそれ(手順書でグラフを無効にした理由) | 試すなら最後 |
| 0.8B、ライブラリあり、グラフあり | 最も速い | 余裕あり(同時稼働が可能) | 要確認 | 本命 |

4B で続ける場合も、段階1(複数ゲームの同時進行)と段階3(高速化ライブラリ)の効果はそのまま使える。

## 8. リスクと戻し方

- コンテナ化がうまくいかない場合は、今の Windows ネイティブ構成に戻す(今の構成は変えずに残しておく)
- 世代の記録(`generations.json`、`gen-XXX/`、`replays/`)は、構成を変えても使い続ける。0.8B の系統は、世代番号を分けるか、`model` の欄で区別する
- 配信は Windows 側で続けるので、学習の構成を変える作業中も配信は止めずに済む
