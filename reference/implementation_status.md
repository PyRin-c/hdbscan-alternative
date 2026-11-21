# HDBSCAN の実装計画

`umap_alternative` の独立実装、環境固定、公開 API による比較、配布方法を引き継ぐ。比較基準は Windows と Linux の CPython 3.13 向け Wheel がある `scikit-learn-contrib/hdbscan` の 0.8.44。利用先で使われている上流バージョンと呼び出し設定は、移行時の比較ケースとして追加する。[参照 Wheel](https://pypi.org/project/hdbscan/0.8.44/)

## 公開仕様から確認した処理

上流は scikit-learn 形式の推定器を公開し、NumPy、SciPy、scikit-learn、joblib に依存する Python パッケージとして配布されている。密・疎の特徴量と、密・疎の距離行列を入力にできる。Rust 側では数値処理を共通のコアに置き、C ABI から Python ランタイムなしで呼べるようにする。[上流リポジトリ](https://github.com/scikit-learn-contrib/hdbscan) · [PyPI の配布情報](https://pypi.org/project/hdbscan/0.8.44/)

1. 指定した距離で各点のコア距離を求め、点間距離と両端のコア距離から相互到達可能距離を計算する。`min_samples` はコア距離、`min_cluster_size` は階層の凝縮に関わる。contrib 版での自己点、同距離の順位、`min_samples=None` の挙動を比較データで確認する。
2. 相互到達可能距離の最小全域木（MST）から、単連結法の階層を作る。公開 API には `generic` と KD-tree・Ball-tree を使う Prim/Borůvka の経路があり、`best` や近似 MST による経路の違いも互換性検証に含める。
3. `min_cluster_size` に従って階層を凝縮し、安定性に基づく EOM（Excess of Mass）または葉を使う `leaf` でクラスタを選ぶ。点ごとのクラスタ番号（ノイズは `-1`）、所属強度、クラスタごとの永続性、各種の木が公開出力になる。[アルゴリズム解説](https://hdbscan.readthedocs.io/en/latest/how_hdbscan_works.html) · [公開 API](https://hdbscan.readthedocs.io/en/latest/api.html)

## 実装の順序

| 段階 | 対象と完了条件 |
| --- | --- |
| 基盤 | Rust ワークスペース、C のロード確認、Python Wheel の隔離インストール、`reference/` の再現確認を用意する。 |
| 最初の学習経路 | 密な有限 `float64` 特徴量、ユークリッド距離、明示的な `algorithm="generic"`、厳密 MST、`cluster_selection_method="eom"`。ノイズを含む入力と重複・同距離を含む入力で階層と公開出力を照合し、同じ Rust コアを Python の `fit` / `fit_predict` と C ABI から使う。 |
| 入力と選択肢 | 密・疎の事前計算距離、疎な特徴量、その他の距離、`leaf`、クラスタ選択の閾値、単一クラスタの許可、欠損値とエラー、既定の `best` と木を用いた各経路。引数・入力形式ごとに比較を追加する。 |
| 学習後の機能 | `outlier_scores_`、`dbscan_clustering`、`approximate_predict`、`approximate_predict_scores`、ソフトクラスタリング、妥当性指標、`RobustSingleLinkage`、`BranchDetector`、木のエクスポートと Python の補助 API。公開属性の生成条件、空の結果、予測の対象外となる設定を観測する。 |
| 配布と高速化 | C は固定幅のステータス、バッファの形状・長さ、モデルと結果の所有権・解放手順を定義する。Python は NumPy・SciPy の入出力と例外を接続する。正確性の比較を維持しながら、同じ入力・設定・スレッド数で実行時間とピークメモリを測る。 |

最初の経路では、手計算できる点集合でコア距離と相互到達可能距離を確認する。次に MST、単連結法の階層、凝縮木、ラベル・所属強度・永続性の順に公開出力と照合する。上流の公開属性で取得できる中間出力は `minimum_spanning_tree_.to_numpy()`、`single_linkage_tree_.to_numpy()`、`condensed_tree_.to_numpy()` で採取する。[公開 API](https://hdbscan.readthedocs.io/en/latest/api.html) · [高速化の論文](https://arxiv.org/abs/1705.07321)

## 参照データと合格条件

- `reference/uv.lock` で上流と実行依存を固定する。比較データには型、形状、設定、OS・アーキテクチャ、Python と依存のバージョンを記録する。最初のケースは BLAS などのスレッドプールを 1 スレッドに制限する。実装側のテストでは保存済みデータを使い、上流 Wheel の実行は専用の CI ジョブで行う。
- 整数は値と順序、浮動小数点は型とビット列で比較する。JSON では浮動小数点を行優先順に並べ、`float64` は16桁、`float32` は8桁の16進文字列で表す。ラベルの付け替えや木の同型判定は差分の診断に使う。クラスタ番号、木のノード ID、同じ重みの辺の順序も、公開出力との直接比較で検証する。
- 空入力・1行・特徴量0、重複、同距離、全点ノイズ、単一クラスタ、`NaN` と無限大、疎な距離グラフの非連結、オプションの境界を機能ごとに追加する。OS や依存ライブラリによって差が出る設定は条件を記録し、対応する参照データを用意する。
- 差分を見つけたら再現するケースを記録し、対象の公開出力がビット単位で一致するまで修正と再比較を続ける。差分が残る機能は検証中として扱う。

## 現在の状態

Minkowski と Chebyshev の予測データ・枝検出データを Python と Rust で生成できるようにした。Python では、上流 Wheel と同じく Minkowski の近傍キャッシュを既定の指数 `2.0` で作る。近傍、予測ラベル、所属確率、外れ値スコア、枝グラフの型と値、枝予測、代表点、例外を Windows の144条件と Linux の142条件で照合した。Linux で上流が異常終了した2条件では、`full` の直前までの出力を別に照合し、本実装の `full` は凝縮木の容量超過を `ValueError` として返した。非連結の枝グラフは、根から到達できる階層を検査してから凝縮する。[予測の参照データ](fixtures/prediction-0.8.44-windows-x86_64.json) · [枝検出の参照データ](fixtures/major-0.8.44-windows-x86_64.json)

非連結の密な事前計算距離で、Python の `HDBSCAN` と関数版が同じ MST・階層の処理を使うようにした。各 OS の232条件を上流 Wheel と隔離プロセスで照合し、結果や例外を取得できた Windows の172条件、Linux の160条件ですべて一致した。学習出力に加え、外れ値スコア、相対妥当性、DBSCAN*、MST 属性へのアクセス時の警告を比較した。上流プロセスが異常終了した Windows の60条件と Linux の72条件では、独立実装が凝縮木の容量を事前に検査し、`ValueError` を返すことを確認した。joblib のキャッシュから学習結果を読み出す場合も、呼び出し時の `numpy.errstate` に従って警告・例外を返す。[Windows の参照データ](fixtures/major-0.8.44-windows-x86_64.json) · [Linux の参照データ](fixtures/major-0.8.44-linux-x86_64.json)

木探索に Chebyshev、Minkowski、標準化ユークリッド、Mahalanobis、Hamming、Canberra、Bray–Curtis、Jaccard、Dice、Rogers–Tanimoto、Russell–Rao、Sokal–Michener、Sokal–Sneath、Haversine を追加した。Chebyshev の KD-tree ではノード半径を公開結果に合わせ、同距離の MST 辺の順序を照合した。Python の `HDBSCAN` は `generic` の Minkowski で指定した指数を使い、木探索では公開版が使用する既定の指数 `2.0` を使う。`RobustSingleLinkage` と Rust・C の距離指定では木探索に指数を渡せる。Windows・Linux の各3,726条件で、ラベル、所属強度、階層、MST、警告、例外の差分は0件だった。[上流の公開 API](https://hdbscan.readthedocs.io/en/latest/api.html)

Python の単連結階層、凝縮木、最小全域木、枝の近似グラフに Matplotlib の描画を追加した。凝縮木のバーの本数、対数幅、枝の座標を Windows・Linux の各240条件で照合し、無限の λ を持つクラスタの描画パラメーターと警告も確認した。scikit-learn の `BaseEstimator` を継承し、`clone`、`Pipeline`、`GridSearchCV`、joblib のキャッシュを確認した。両 OS で C ABI の動作確認と、隔離環境にインストールした Wheel の264テストが通過した。[描画の公開 API](https://hdbscan.readthedocs.io/en/latest/api.html#utilities) · [scikit-learn の推定器インターフェース](https://scikit-learn.org/stable/developers/develop.html)

Iris、Wine、Breast Cancer Wisconsin、Digits の4データセットで、Windows・Linux の各176呼び出しの出力を照合した。学習ラベル、所属強度、木、外れ値スコア、相対妥当性、近似予測と `RobustSingleLinkage` を比較し、配列の型・形状・バイト列の差分は0件だった。`reference/` のスクリプト、ロックファイルと比較データは `umap_alternative` と同様に Git 管理対象とし、`reference/.venv/` はローカルに置く。利用先固有の入力・設定の照合は実データの指定後に行う。[公開データセット](https://scikit-learn.org/stable/datasets/toy_dataset.html) · [実データ比較スクリプト](compare_public_datasets.py)

`RobustSingleLinkage` と `robust_single_linkage` を追加した。密なユークリッド・Manhattan 特徴量では Prim、Borůvka、`best` と `generic` を利用できる。Python の `generic` は密な事前計算距離、疎な特徴量、追加の距離関数にも対応する。階層に入る辺の順序は経路ごとに公開出力を照合し、Ball-tree の距離への `alpha` 適用による丸めも再現する。Windows・Linux の各7,680条件で、比較したラベル、階層、警告、例外の差分は0件だった。[上流の公開 API](https://hdbscan.readthedocs.io/en/latest/api.html#robustsinglelinkage)

`BranchDetector`、`detect_branches_in_clusters`、`approximate_predict_branch` を追加した。密なユークリッド・Manhattan 特徴量で `core` と `full` のグラフを構築する。手動指定のクラスタラベル、欠損行、枝の永続性、代表点、枝の凝縮木と単連結階層を扱う。Windows・Linux の各482条件で、近傍キャッシュ、枝検出、グラフ、予測、代表点、警告、例外の差分は0件だった。重み付きメドイドは各行の座標差から距離を求め、評価値が等しいときに最初の行を選ぶ。両機能の参照データを CI に登録し、C ABI の学習・枝検出・近似予測と、隔離環境での Wheel のテストを両 OS で確認した。枝検出と近似予測の距離は密なユークリッドと Manhattan を対象とする。[上流の公開 API](https://hdbscan.readthedocs.io/en/latest/api.html#branch-detection) · [Windows の参照データ](fixtures/major-0.8.44-windows-x86_64.json) · [Linux の参照データ](fixtures/major-0.8.44-linux-x86_64.json)

枝検出の追加検証では、198入力を一つずつ別プロセスで実行した。Windows・Linux の各978呼び出しで、取得できた出力、警告、例外が一致した。空の全域森、連結階層に同じ子孫が現れる場合の凝縮木の容量、枝ラベルを元の行に割り当てるときの形状エラーを照合した。凝縮木の容量は階層を展開する前に計算し、子の ID が親より前にあることを検査する。上流 Wheel では2入力の `full` を `label_sides_as_branches` の両設定で8秒まで待ったが、計4呼び出しで結果が得られなかった。独立実装ではこれらを `ValueError` として返す。Linux では別の1入力で、記録した例外を返した後に上流プロセスが異常終了した。これら3入力で学習時のラベルを明示的に渡す経路は、両 OS の各6条件で一致した。

呼び出し側が所属強度を指定した場合は、枝の所属強度との合成に元の配列の型を使う。`float32` の丸めと整数配列への書き戻しで生じる型エラーを再現し、欠損行がある場合は有限な行の結果を元の位置に戻す。Windows・Linux の各360条件で一致した。再現ケースを参照データに追加し、両 OS で C ABI の動作と隔離環境に入れた Wheel の254テストを確認した。

密・疎のユークリッド・Manhattan・cosine 特徴量と事前計算距離の `generic`、密なユークリッド・Manhattan の Prim と Borůvka、EOM と `leaf`、単一クラスタの許可、クラスタ選択閾値、欠損行、GLOSH と DBSCAN* を Rust、Python、C ABI に接続した。各 OS の89ケースで、ラベル、所属強度、永続性、単連結階層、凝縮木、GLOSH、相対妥当性スコアと各ケース6通りの DBSCAN* カットを比較し、公開 API で取得できる MST の辺も照合する。C は各入力経路の学習から結果取得と解放までを確認する。[Windows の参照データ](fixtures/hdbscan-0.8.44-windows-x86_64.json) · [Linux の参照データ](fixtures/hdbscan-0.8.44-linux-x86_64.json)

密なユークリッド・Manhattan 特徴量で `approximate_predict` と `approximate_predict_scores` を Rust・Python・C ABI から利用できる。学習行の近傍距離、同距離の候補の順序、凝縮木上の遡り条件、クラスタの代表点を公開 Wheel で確認した。Windows・Linux の各14参照ケースと400追加ケースで、予測ラベル、接続先、所属強度、外れ値スコアを比較した。列優先や逆向きの特徴量を含むユークリッド距離の `generic` 学習では、列のストライドに応じたノルムの加算順で照合した。残る学習後の機能と利用先の実データは引き続き検証対象とする。[Windows の予測参照データ](fixtures/prediction-0.8.44-windows-x86_64.json) · [Linux の予測参照データ](fixtures/prediction-0.8.44-linux-x86_64.json)

同じ推定器を再学習して予測データを作り直さない場合は、最新の凝縮木で接続先の離脱 λ を調べ、キャッシュのクラスタ間の枝を遡る。外れ値スコアでは、最も近い学習点までの距離が `0.0` なら、相互到達可能距離で選ばれた接続先の離脱 λ を使う。再学習で行数が減り、以前の行番号が新しい木のクラスタ ID になった場合も含め、Windows・Linux の各192条件で公開版と一致した。

`membership_vector` と `all_points_membership_vectors` を Rust・Python・C ABI に追加した。両 API は最も近い代表点への距離と、凝縮木上でクラスタに合流する λ を組み合わせる。新規点の外れ値に基づく重みでは点の λ を `float32` に丸め、学習点の重みでは行列全体の最大値を使って指数関数の入力を調整する。列優先の学習点と新規点について、代表点までの距離を公開版が使うメモリ順で照合した。選択クラスタが空のとき、Python の学習点 API は一次元のゼロ配列を返す。凝縮木にクラスタ間の枝がなく根が選ばれた場合は `KeyError` を返す。[上流のソフトクラスタリング解説](https://hdbscan.readthedocs.io/en/latest/soft_clustering_explanation.html)

Windows・Linux の各14参照ケース、464追加条件、配列の並び方を変えた184追加条件で、ソフトクラスタリングのビット列、警告、例外を比較した。C ABI は各93条件を追加比較し、129・163クラスタを含む行列まで照合した。重複点で現れる無限大の λ、`NaN` の符号、極小の所属確率も比較した。予測キャッシュを残した再学習では、上流 Wheel が応答した各45条件で一致した。応答が得られなかった各19条件は照合対象外とし、Rust 側は利用可能な密度や枝がない場合にエラーを返す。

`validity_index` の全体スコアとクラスタ別スコアを Python・Rust・C ABI から取得できる。クラスタ内の全点コア距離、最小全域木の内部点と辺、クラスタ間の最小密度分離を公開 API で照合する。クラスタ内とクラスタ間の距離の丸め順や、同じ重みの辺をたどる順番も比較対象とした。各 OS の参照データは30のスコアケースと13の補助関数ケースで構成する。[Windows の参照データ](fixtures/validity-0.8.44-windows-x86_64.json) · [Linux の参照データ](fixtures/validity-0.8.44-linux-x86_64.json) · [DBCV 原論文](https://www.dbs.ifi.lmu.de/~zimek/publications/SDM2014/DBCV.pdf)

Windows・Linux の各環境で Python API の5,324追加条件を比較し、実在するクラスタと全体のスコア、補助関数の数値、警告の種類名と本文、例外が公開版と一致した。C ABI は各2,668追加条件で取得できる数値をビット単位で照合し、異常入力は返却ステータスで確認した。新規にビルドした Wheel のみをインストールした環境でも各231件のテストが通った。

クラスタ番号に欠番がある場合、その位置の公開版の値は実行間で変わる。本実装は欠番に `NaN` を返す。Python の `n_jobs` による距離行列の列分割は公開版に合わせ、計算は列ブロック順に行う。公開版で欠損距離の `copy=False` と複数ワーカーを組み合わせた場合の数値や、複数ブロックが不正なノルム配列を受けた場合の例外文面には、実行順に応じた違いがある。

`relative_validity_` は学習時の MST からクラスタ内の最大距離とクラスタ間の最小距離を取り出す。クラスタ間の辺がない場合は木の最大距離を基準にする。単一クラスタでノイズへの辺があれば、その最小距離を使う。欠損行のある学習結果では公開版が返す MST の点番号を使い、分母にはノイズを含む全行数を使う。Python は値を初回の読み出し後に保持し、再学習後も同じ値を返す。`None` を代入すると次回に再計算する。CSR 事前計算距離で木に両端がクラスタに属する辺がある場合は、公開版の例外を Python で再現し、C は `INVALID_SHAPE` を返す。[上流の公開 API](https://hdbscan.readthedocs.io/en/latest/api.html#hdbscan.hdbscan_.HDBSCAN)

Windows・Linux の各環境で、1,970通りの学習条件と、うち1,959通りのラベル差し替えを追加で照合した。Python が返すスコアのビット列、警告の種類名と本文、例外が一致した。C ABI でも同じ1,970通りの学習条件において、取得できるスコアのビット列またはエラーに対応するステータスが一致した。新規ビルドした Wheel を隔離環境に入れ、各235テストの通過を確認した。

Prim のユークリッド距離は列順に差の二乗を足して平方根を取り、Manhattan は差の絶対値を列順に足す。同じ重みの候補は先に見つかった接続元を保持し、次の点は入力行番号の小さい方を選ぶ。公開 MST は単連結階層を作る順序で返す。距離が `float64` の最大値に達した場合の自己ループ、凝縮木の枝の重複、配列容量を超えた場合の例外も照合する。[上流の学習オプション](https://hdbscan.readthedocs.io/en/latest/api.html#hdbscan)

Python の既定 `best` は疎入力と事前計算距離で `generic`、密なユークリッド・Manhattan 特徴量は60列以下で Borůvka の KD-tree、61列以上で Prim を選ぶ。Borůvka の近傍探索は同距離時の順序も照合する。16,384行を超え複数ワーカーを使う場合は、問い合わせを連続した行に分け、元の行順で結果を連結する。近似モードは成分が減る間、探索範囲の上限を次の巡回に引き継ぐ。Ball-tree は親・子ノードの半径差を使って上限を絞り込む。厳密モードでも、その結果として上流がより重い辺を返す入力を参照ケースに含めた。[高速化の論文](https://arxiv.org/abs/1705.07321) · [上流の Ball-tree 報告](https://github.com/scikit-learn-contrib/hdbscan/issues/729)

Windows・Linux の各環境で、Borůvka の厳密・近似と `best` の比較2,400ケース、既定設定の比較108ケース、16,384行・16,389行の並列設定24ケース、Prim と `best` の比較1,172ケース、Prim の距離上限付近90ケースを追加で照合した。各比較で取得できる公開数値出力、警告、例外の差分は0件だった。

GLOSH は各点の離脱時の λ と、所属したクラスタの子孫までの最大 λ を使い、`(最大 λ − 離脱時の λ) / 最大 λ` を計算する。離脱時の λ が非有限、または最大 λ が `0.0` なら `0.0` を返す。有限の λ と無限大の最大 λ の組み合わせでは公開 Wheel の `NaN` のビット列も照合する。Python では同じ推定器で再学習しても取得済みの配列を保持する。DBSCAN* は境界距離未満の結合を順に行い、最小サイズを満たす成分に代表ノード番号順でラベルを振る。追加比較では、各 OS の349件のスコア、29,930通りのカット、492通りの引数変換で、数値のビット列、ラベル番号、例外の型とメッセージが上流と一致した。[上流の外れ値検出](https://hdbscan.readthedocs.io/en/latest/outlier_detection.html) · [上流の DBSCAN* API](https://hdbscan.readthedocs.io/en/latest/api.html#hdbscan.hdbscan_.HDBSCAN.dbscan_clustering)

EOM は単一クラスタを許可した場合に根も候補に含め、根と子孫の選択結果の安定性が等しい場合は根を選ぶ。距離の下限が `0.0` で根を選んだときは、根から出る辺の最大 λ を下回る点をノイズにする。この場合、上流では永続性が `1.0` を超えることがある。`leaf` は末端の非根クラスタを選び、凝縮木に根しかなければ全点をノイズにする。Windows の追加比較では、固定シードの整数・正規分布データを使った密・疎のユークリッド入力、6・17・30点、1・3次元、3通りの選択設定で比較し、2,592ケースの6種類の出力がすべてビット一致した。[上流のパラメーター解説](https://hdbscan.readthedocs.io/en/latest/parameter_selection.html#leaf-clustering)

`cluster_selection_epsilon` と等しい距離で始まった候補は残る。下の枝から親をたどる場合は、距離が閾値と等しい親を通過する。EOM の距離・サイズ上限の適用後に距離の下限による統合を行い、上限を超えた親が選ばれる場合がある。距離の下限を指定して根だけが選ばれた場合、点の離脱 λ が `1 / cluster_selection_epsilon` 以上ならその点は根に属する。EOM は各子の安定性を凝縮木の順に足し、加算ごとに `float32` に丸める。[上流の API](https://hdbscan.readthedocs.io/en/latest/api.html#hdbscan)

`cluster_selection_persistence` を正にすると、凝縮木の λ と閾値を `float32` に丸め、クラスタ選択前に枝を統合する。短命な枝とその兄弟の統合、深い分岐が残る場合の付け替え、分岐がない場合の `IndexError` を照合した。Windows と Linux の追加比較では、各6,300ケースの密・疎入力、2,400ケースの階層距離行列、512ケースの極端な尺度で差分が0件だった。取得できる数値出力をビットで比較し、例外の型とメッセージ、極端な尺度での NumPy 警告も照合した。

重複と同距離のケースでは、ラベル `1` で所属強度 `0.0` の点がある。全点ノイズのケースでは永続性の配列が空になる。

ユークリッド距離の行ノルムは、8要素ごとの偶数列・奇数列の積をそれぞれ後ろから足し、端数を前から加える。8列・16列で丸め順を検証する参照ケースを追加した。Windows では `numpy.random.default_rng(956)` で標準偏差 `0.95`、平均 `0`、`3`、`8` の正規分布から各10行を生成し、特徴量1〜8、16、9、15、17、24、32、64列の順に各25ケースを比較した。375ケースで全出力がビット一致した。

疎なユークリッド特徴量のノルムは格納順に値の二乗を加算する。内積は左行の格納順に走査し、同じ列の右行の値を格納順に掛け合わせる。参照の非整列・重複列、空行、未格納のゼロ、丸めを比較する。Windows の追加比較は6・18・64行、9・64・512列の整列済み、非整列、重複列あり、整数入力の計180ケースで、6種類の出力がすべてビット一致した。Python の `generic` のユークリッドと cosine は `float32` 特徴量を受け取ると、上流と同じバッファ型の `ValueError` を返す。[ユークリッド距離の参照式](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.euclidean_distances.html)

特徴量の `NaN` と正負の無限大を含む行を除外し、ラベル `-1`、所属強度 `0.0` を元の行位置に戻す。凝縮木には離脱 λ `0.0` の点を先頭に追加し、単連結階層には距離 `+inf` の結合を末尾に追加する。追加順は参照環境の CPython の行番号集合の順序で、学習に残る点が2点のときの連結先番号も公開出力に合わせる。MST は学習に残った行の連番を返す。非有限値を含む CSR では重複列を内部コピーで合算してから有効行を判定する。各 OS の追加比較は特徴量6,642ケース、重複列を持つ CSR 876ケース、事前計算距離975ケースで、取得できる数値出力のビット列、警告、例外の型とメッセージが一致した。

密な事前計算距離は、有限の対称行列、欠損のある連結行列、非対称かつ対角が非ゼロの行列で比較する。上流の公開関数 `hdbscan.hdbscan()` が返す MST の辺を Rust の出力と照合する。Python は、上流の推定器と同様に事前計算距離で `minimum_spanning_tree_` を参照された場合に警告と `None` を返す。有限の相互到達可能距離で全点が結ばれない行列は、Rust・Python・C でエラーを返す。

Manhattan の CSR 経路は、列が非整列のときに SciPy の公開 `sort_indices()` で観測した同順位の並びを再現する。短い行では安定な順序、長い行では区分後の順序に従って重複を合算し、区分の上限に達した場合に使われるヒープの同順位の並びも照合する。CSR の `float32` は列を合算し、両行に格納された値の差を `float32` で計算してから `float64` に積算する。cosine は行を正規化して内積から距離を求める。密入力ではノルムが `10 × f64::EPSILON` 未満、CSR ではノルムがゼロのときに元の行の値を使う。CSR cosine の内積は重複列と格納順を使う。Windows で5・17・65特徴量の密入力、疎入力、重複を含む非整列 CSR、`float32` の Manhattan を追加比較し、324ケースで6種類の出力がすべてビット一致した。[SciPy の CSR 仕様](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html) · [cosine 距離](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_distances.html)

疎な事前計算距離は、重複列を加算した行の格納値でコア距離を求める。明示的なゼロと対角値は近傍順位に含み、MST の候補は非対角かつスケーリング後に非ゼロの辺から作る。`alpha` と `max_dist` は上流の観測に合わせて `float32` に丸めてから使う。同じ重みでは CSR の行・列順で候補を選び、MST を行・列順に戻してから階層作成用に並べ替える。疎行列そのものの非連結と、相互到達可能距離を計算した後の非連結は別のエラーになる。Windows の追加比較では、対称・非対称の疎グラフ計144ケースのうち、両実装で学習できた126ケースの Python 公開出力がすべてビット一致し、18ケースで両方が `ValueError` を返した。[SciPy の MST 仕様](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csgraph.minimum_spanning_tree.html)

MST の同じ重みの辺は、NumPy 2.5.5 の `X86_V3` の `argsort` から観測した順に並べる。257本以上は16要素のブロックで分割し、256本以下の区間をソートネットワークで整列する。分割回数の上限に達した場合の同順位の順序は Windows と Linux の公開出力に合わせる。256本・257本・259本の密な特徴量と321本の疎な事前計算距離を参照ケースに追加した。Windows と Linux の追加比較では、固定シード `515816148` で256〜1,026点の密・疎のユークリッド・Manhattan 特徴量と事前計算距離を生成し、各972ケースで取得できる公開出力がすべてビット一致した。Linux の参照データ生成では `X86_V3` を使用するよう NumPy の AVX-512 向け処理を無効にする。[NumPy の `argsort` 仕様](https://numpy.org/doc/stable/reference/generated/numpy.argsort.html) · [CPU 別の実行経路](https://numpy.org/doc/stable/reference/simd/build-options.html#runtime-dispatch) · [ブロック分割の論文](https://arxiv.org/abs/1704.08579)
