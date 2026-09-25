# Kevローカルサーバー運用ガイド

この文書は、Windows PC上にデプロイしたKevをローカルAPIとして利用するための手順書です。

## 現在の構成

| 項目 | 値 |
|---|---|
| Kev本体 | `D:\GitHub\kev` |
| API URL | `http://127.0.0.1:8009` |
| モデル | `jaredpalmer/kev-4b` |
| APIモデル名 | `kev-latest` または `jev-latest` |
| デバイス | NVIDIA GeForce RTX 5060 Ti(VRAM 16GB)/ CUDA 12.8 |
| PyTorch | 2.8.0+cu128 |
| Python | 3.12 |
| バックエンド | PyTorch / bfloat16 |
| CUDAグラフ | 無効（16GB VRAMでの安定性を優先） |
| 認証 | 現在はなし。ローカルホストからのみ接続可能 |

Kevは文章生成モデルではなく、状態と質問を受け取り、候補の選択・Yes確率・段階評価を返すJev互換の意思決定モデルです。

## 稼働確認

PowerShellで次を実行します。

```powershell
Invoke-RestMethod http://127.0.0.1:8009/v1/models |
    ConvertTo-Json -Depth 8
```

レスポンスの `device` が `cuda`、`backend` が `torch` ならGPUで稼働しています。

ポートだけを確認する場合:

```powershell
Get-NetTCPConnection -LocalPort 8009 -State Listen
```

## 起動

通常はフォアグラウンドで起動するとログを直接確認できます。

```powershell
cd D:\GitHub\kev
$env:HF_HUB_DISABLE_SYMLINKS = "1"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$env:KEV_CUDA_GRAPHS = "0"
.\.venv\Scripts\python.exe -m kev.serve `
    --run jaredpalmer/kev-4b `
    --port 8009
```

初回起動ではHugging FaceからKevのチェックポイントとQwen3.5のベースモデルをダウンロードします。`Uvicorn running on http://127.0.0.1:8009` と表示されたら準備完了です。

バックグラウンドで起動する場合:

```powershell
cd D:\GitHub\kev
$env:HF_HUB_DISABLE_SYMLINKS = "1"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$env:KEV_CUDA_GRAPHS = "0"

Start-Process `
    -FilePath "D:\GitHub\kev\.venv\Scripts\python.exe" `
    -ArgumentList @(
        "-m", "kev.serve",
        "--run", "jaredpalmer/kev-4b",
        "--port", "8009"
    ) `
    -WorkingDirectory "D:\GitHub\kev" `
    -RedirectStandardOutput "D:\GitHub\kev\kev-4b-server.out.log" `
    -RedirectStandardError "D:\GitHub\kev\kev-4b-server.err.log" `
    -WindowStyle Hidden
```

バックグラウンド版のログ:

```powershell
Get-Content D:\GitHub\kev\kev-4b-server.err.log -Tail 50 -Wait
```

## 停止

8009番ポートを待ち受けているKevプロセスだけを停止します。

```powershell
$listener = Get-NetTCPConnection -LocalPort 8009 -State Listen `
    -ErrorAction SilentlyContinue

if ($listener) {
    Stop-Process -Id $listener.OwningProcess
}
```

## Choice APIを使う

次の例では、現在の状態に対して2つの行動候補から1つを選ばせます。

```powershell
$body = @{
    state = "The build failed because unit tests do not pass."
    model = "kev-latest"
    questions = @{
        action = @{
            type = "choice"
            instructions = "What should happen next?"
            criteria = @{
                fix_tests = "Investigate and fix the failing tests"
                deploy = "Deploy immediately without testing"
            }
        }
    }
} | ConvertTo-Json -Depth 8

Invoke-RestMethod `
    -Uri http://127.0.0.1:8009/v1/systemone `
    -Method Post `
    -ContentType application/json `
    -Body $body |
    ConvertTo-Json -Depth 8
```

レスポンス例:

```json
{
  "model": "kev-latest",
  "answers": {
    "action": {
      "type": "choice",
      "choice": "fix_tests",
      "confidence": 0.874,
      "probabilities": {
        "deploy": 0.063,
        "fix_tests": 0.937
      }
    }
  },
  "latency_ms": 265.2
}
```

`choice` が選ばれた候補、`probabilities` が各候補の確率です。`confidence` は正解率ではないため、重要な処理では独自データで閾値を評価してください。

## Pythonから使う

追加ライブラリを使わない例です。

```python
import json
import urllib.request

payload = {
    "state": "The build failed because unit tests do not pass.",
    "model": "kev-latest",
    "questions": {
        "action": {
            "type": "choice",
            "instructions": "What should happen next?",
            "criteria": {
                "fix_tests": "Investigate and fix the failing tests",
                "deploy": "Deploy immediately without testing",
            },
        }
    },
}

request = urllib.request.Request(
    "http://127.0.0.1:8009/v1/systemone",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)

with urllib.request.urlopen(request, timeout=30) as response:
    result = json.load(response)

print(result["answers"]["action"]["choice"])
print(result["answers"]["action"]["probabilities"])
```

## テトリスから使う

`kev-tetris` はこのデプロイ(`D:\GitHub\kev\.venv`)を使ってKevを起動します(`KEV_CUDA_GRAPHS=0` などの環境変数も同じ設定で渡します)。配信画面と学習コントロールは次で起動します。

```powershell
cd D:\GitHub\kev-tetris
python -m kev_tetris.stream
```

配信画面は `http://127.0.0.1:8765`、学習コントロール(学習開始・一時停止・再開・停止)は `http://127.0.0.1:8765/control` です。

| ポート | 用途 |
|---|---|
| 8009 | この手順書で手動起動するKev。配信画面は、同じモデルを提供していればこれを再利用する |
| 8011 / 8012 | 配信画面が起動するKev |
| 8019 | 強化学習ループが起動するKev |

VRAM 16GBではKev-4Bを同時に2つ載せられません。**強化学習を始める前に、8009番のKevを「停止」の手順で止めてください。** 止めていない間、学習コントロールには「GPUの空き待ち」と表示され、学習は待機します。学習中の配信画面は、各世代のテスト時のプレイを録画再生します。

## APIキーを設定する場合

起動前に `KEV_API_KEY` を設定すると `/v1/*` にBearer認証が必要になります。

```powershell
$env:KEV_API_KEY = "replace-with-a-long-random-secret"
```

クライアント側では次のヘッダーを追加します。

```text
Authorization: Bearer replace-with-a-long-random-secret
```

APIキーを設定してもサーバーは現在と同じく `127.0.0.1` にバインドします。LANやインターネットへ公開する場合は、別途リバースプロキシ、TLS、ファイアウォール設定が必要です。

## 環境を作り直す

Windows版CUDA PyTorchとの互換性のため、このデプロイではPython 3.12を使います。

```powershell
cd D:\GitHub\kev
uv venv --python 3.12 --clear
uv sync --extra serve --python 3.12
uv pip install `
    --python .venv\Scripts\python.exe `
    --reinstall "torch==2.8.0" `
    --index-url https://download.pytorch.org/whl/cu128
```

CUDA確認:

```powershell
.\.venv\Scripts\python.exe -c `
    "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

期待値は `2.8.0+cu128`、`True`、`NVIDIA GeForce RTX 5060 Ti` です。

## トラブルシューティング

### シンボリックリンク権限エラー

`WinError 1314` が出る場合は、起動前にコピー方式を指定します。

```powershell
$env:HF_HUB_DISABLE_SYMLINKS = "1"
```

### 接続できない

```powershell
Get-NetTCPConnection -LocalPort 8009 -State Listen
Get-Content D:\GitHub\kev\kev-4b-server.err.log -Tail 100
```

待受プロセスがなければ「起動」の手順を再実行します。

### `cuda_available False`

CPU版PyTorchへ置き換わっています。「環境を作り直す」のCUDA 12.8版インストールを実行してください。

### ポートが使用中

```powershell
Get-NetTCPConnection -LocalPort 8009 -State Listen |
    Select-Object LocalAddress, LocalPort, OwningProcess
```

既存のKevを使うか、「停止」の手順で終了してから再起動します。
