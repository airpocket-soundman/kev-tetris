# kev-tetris

ローカルの決定モデル [Kev](https://github.com/jaredpalmer/kev) にテトリスをプレイさせ、強化学習で上達させ、世代を入れ替えながら生配信するためのプロジェクト。

全体の計画と経過は [docs/plan.md](docs/plan.md)、高速化の構想は [docs/speedup-plan.md](docs/speedup-plan.md) にまとめている。

## 動作環境

| 項目 | 値 |
|---|---|
| GPU | **NVIDIA GeForce RTX 5060 Ti(VRAM 16GB)** |
| OS | Windows 11 |
| Kev | `D:\GitHub\kev` にデプロイ済み(Python 3.12 / PyTorch 2.8.0+cu128)。手順は [docs/kev-local-deployment.md](docs/kev-local-deployment.md) |
| モデル | Kev-4B(`jaredpalmer/kev-4b`、ベース `Qwen/Qwen3.5-4B-Base`) |
| kev-tetris | Python 3.12 以上と numpy(Kev 本体は Kev 側の venv で動かす) |

### 16GB で Kev-4B を扱うときの注意

Kev-4B は bf16 で約 9〜10GB を使う。16GB では **4B を同時に2つ載せられず、学習(`kev.train`)中にほかの 4B も載せられない**。そのため次のように動く。

- 学習ループは、GPU の空きが足りない間は「GPUの空き待ち」と表示して待つ(`--need_serve_gb 10` / `--need_train_gb 13`)。手動で起動した 8009 番の Kev は自動では止めないので、学習前に止める
- 配信画面は、学習中は GPU を手放し、各世代のテスト時のプレイを**録画再生**する。学習していない間は**ライブ**でプレイさせる
- 次世代の先読み込み(`--preload`)は既定でオフ

Kev-0.8B に戻す場合は `--start jaredpalmer/kev-0.8b --base Qwen/Qwen3.5-0.8B-Base --model_name Kev-0.8B --need_serve_gb 3 --need_train_gb 6`(配信側は `--start jaredpalmer/kev-0.8b --model_name Kev-0.8B --need_serve_gb 3 --preload 1`)。

### ポート

| ポート | 用途 |
|---|---|
| 8009 | 手動で起動した Kev(配信画面は、同じモデルならこれを再利用する) |
| 8011 / 8012 | 配信画面が起動する Kev |
| 8019 | 学習ループが起動する Kev |
| 8765 | 配信画面(`/`)と学習コントロール(`/control`) |

## しくみ

| ファイル | 役割 |
|---|---|
| `kev_tetris/tetris.py` | テトリス本体(10×20、7-bag)。1手 = 「回転+列」を選んでハードドロップ |
| `kev_tetris/interface.py` | Kev 向けインターフェース。盤面をテキストの `state` に、置き場所の候補を `move` という Choice 質問(1候補ごとに「消えるライン・穴・高さ・凸凹」の説明つき)に変換する |
| `kev_tetris/policy.py` | Kev の `/v1/systemone` を呼ぶ `KevPolicy`、`kev.serve` を起動・停止する `KevServer` |
| `kev_tetris/kevenv.py` | デプロイ済み Kev の場所(`KEV_HOME`、`KEV_PYTHON` で変更可) |
| `kev_tetris/rl.py` | 強化学習ループ(下記)。一時停止・停止に対応 |
| `kev_tetris/replays.py` | 各世代のテストのプレイの記録と再生 |
| `kev_tetris/stream.py` + `static/` | 配信画面と学習コントロール |

### 強化学習(1世代分)

1. **練習試合**: 現世代の Kev が、自分の出した確率に従って手をサンプリングしながら16ゲーム遊ぶ(探索)
2. **評価**: 各手に報酬(ライン消去 +、穴・積み上げ −、ゲームオーバー −10)を与えて割引リターンを計算し、盤面特徴の線形ベースラインとの差を「アドバンテージ」とする
3. **学習**: 期待より良かった手(アドバンテージ上位)を正解ラベルにして、前世代から `kev.train --init_from` で追加学習する(advantage-filtered な方策改善 / expert iteration)
4. **テスト**: 固定シードで貪欲にプレイし、平均ライン数を `runs/generations.json` に、プレイを `runs/replays/` に記録する

## 使い方

配信画面と学習コントロールを起動する:

```bash
python -m kev_tetris.stream
```

- 配信画面: http://127.0.0.1:8765/ (1920×1080。OBS のブラウザソース用)
- 学習コントロール: http://127.0.0.1:8765/control (**学習開始・一時停止・再開・停止**のボタン。配信には映らない)

学習コントロールを使わず、ターミナルから学習を実行する場合(コントロールページからも状態が見え、操作できる):

```bash
python -m kev_tetris.rl --generations 0
```

`--generations 0` は「停止するまで世代を重ね続ける」(操作ページの「学習開始」の既定)。`--generations 10` のように数を指定すると、その世代で終了する。

- 一時停止は1手ごとの区切りで止まる。`kev.train` の実行中はそのプロセスごと一時停止する(GPU メモリは確保したまま)
- 停止すると途中の世代は捨てられ、次に開始したときその世代からやり直す
- ログ: `runs/rl.log`、学習ログ: `runs/gen-XXX.train.log`

配信画面のオプション:

- `--source auto|live|replay`: ライブ / 録画再生 / 自動(既定)
- `--seed 1`: ライブ時、全世代に同じブロック順を与えて公平に比較する
- `--only 0,5,10`: 見せる世代を絞る
- `--pieces_per_gen 150`: 1世代の持ち時間(手数)
- `--demo`: Kev を使わない代役(ヒューリスティック)で画面だけ確認する

## X(旧 Twitter)でライブ配信する

配信画面はブラウザのページなので、OBS Studio で取り込んで X に RTMP で送る。

1. [OBS Studio](https://obsproject.com/) をインストールする
2. OBS の「ソース」→「+」→「ブラウザ」を追加し、URL に `http://127.0.0.1:8765/`、幅 `1920`、高さ `1080` を設定する
3. OBS の「設定」→「配信」で、サービスを「カスタム」にする。サーバーには X の画面にある `RTMPS` の URL(例: `rtmps://jp.pscp.tv:443/x`)を、ストリームキーには X の「Stream key」を入れる
4. OBS の「設定」→「出力」と「映像」を次のように設定する
   - エンコーダ: NVIDIA NVENC H.264(RTX 5060 Ti の専用エンコーダなので Kev の計算とほぼ競合しない)
   - 解像度: 1280×720 または 1920×1080、30fps
   - ビットレート: 4500〜6000 Kbps(CBR)、キーフレーム間隔: 3秒、音声: AAC 128Kbps
5. OBS で「配信開始」を押す。X の作成画面のプレビューに映像が出たら、X 側の「ライブ放送する」を押す

ストリームキーは配信の乗っ取りに使えるので、スクリーンショットや録画に映さないこと。X が推奨する最新の設定値は X のヘルプで確認すること。
