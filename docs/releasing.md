# GitHub Releases の準備

プレビューは GitHub Releases から配布します。配布物は Windows x64 と Linux x86-64 向けの CPython 3.10〜3.15 の Wheel、Windows x64 の C ABI ZIP、`SHA256SUMS` です。

1. `Cargo.toml` のワークスペース版と `docs/releases/v<version>.md` のリリースノートを揃えます。
2. 配布するコミットを GitHub に反映し、そのコミットの CI の通過を確認します。
3. `v<version>` タグを送ると [Prepare GitHub Release](../.github/workflows/release.yml) が動きます。
4. 成果物を添付したドラフトが作られたら、添付ファイルと説明を確認してプレビューを公開します。

ワークフローは対象コミットのテストを実行し、各 Wheel を隔離環境にインストールして確認します。Windows では ZIP を展開し、同梱の学習サンプルと C ABI のスモークテストを実行します。Rust のビルドパスと PDB 参照を配布用に整え、添付前にアーカイブの内容とビルド端末のパスを検査します。

既存タグの再ビルドには、Actions の **Prepare GitHub Release** を手動実行してタグ名を渡します。すでにドラフトがある場合は、同じ実行の成果物と照合してから添付を整えます。

Windows で C ABI ZIP を確認するには、リポジトリのルートで以下を実行します。

```powershell
uv run --no-sync python scripts/release_build.py cargo build --release --locked -p hdbscan-capi
./scripts/package-windows-cabi.ps1
```

ZIP の出力先は `target/cabi-release/` です。
