# Windows x64 C ABI

GitHub Releases の `hdbscan-alternative-cabi-<version>-win-x64.zip` には、DLL、MSVC 用インポートライブラリ、C ヘッダー、学習サンプル、MIT ライセンスが入っています。[GitHub Releases](https://github.com/PyRin-c/hdbscan-alternative/releases)

ZIP を展開し、x64 Native Tools Command Prompt for Visual Studio で同梱サンプルをビルドします。

```bat
cl /nologo /std:c11 /W4 fit_clusters.c hdbscan_capi.lib
fit_clusters.exe
```

アプリでは `hdbscan_capi.h` をインクルードして `hdbscan_capi.lib` をリンクし、実行ファイルと同じディレクトリに `hdbscan_capi.dll` を配置します。DLL の実行には Microsoft Visual C++ の x64 ランタイムを使います。[Microsoft Visual C++ Redistributable](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist/)

密行列は行優先の `double` 配列を渡します。結果ビューの配列は `hdbscan_capi_result_free` まで有効です。予測・枝検出を利用する場合の入力、出力、解放方法は同梱のヘッダーに記載しています。
