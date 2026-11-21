# hdbscan-alternative

[`scikit-learn-contrib/hdbscan`](https://github.com/scikit-learn-contrib/hdbscan) の公開 API と数値結果を基準に、HDBSCAN を Rust で独立実装するプロジェクトです。Rust ライブラリ、他言語から Python ランタイムなしで呼び出せる C ABI、PyO3 と Maturin でビルドする Python Wheel として提供します。

密・疎の特徴量に対するユークリッド・Manhattan・cosine 距離と、密・疎の事前計算距離を使った厳密 MST のクラスタリングを Rust、C、Python から利用できます。クラスタ選択は EOM と `leaf` に対応します。実装順序と互換性の検証範囲は [実装計画](reference/implementation_status.md) に記録しています。

## 配布物

プレビュー版は [GitHub Releases](https://github.com/PyRin-c/hdbscan-alternative/releases) から配布します。

| 配布物 | 対応環境 |
| --- | --- |
| Python Wheel | CPython 3.10〜3.15。Windows x64 または Linux x86-64（glibc 2.34 以上） |
| C ABI ZIP | Windows x64。DLL、MSVC 用ライブラリ、C ヘッダー、学習サンプル |
| `SHA256SUMS` | 配布ファイルの SHA-256 |

Python では対象環境の Wheel を取得し、`python -m pip install <Wheelのファイル名>` でインストールします。C/C++ からの組み込みは [C ABI の使い方](crates/hdbscan-capi/README.md)、配布物の作成は [リリース手順](docs/releasing.md) を参照してください。

## リポジトリの構成

| 場所 | 役割 |
| --- | --- |
| `crates/hdbscan-core` | Rust の数値処理と公開 API |
| `crates/hdbscan-capi` | C ABI の共有ライブラリと公開ヘッダー |
| `crates/hdbscan-python` / `python/hdbscan_alternative` | PyO3 拡張と Python パッケージ |
| `reference/` | 上流 Wheel の固定環境と公開 API の比較データ |

## 開発環境

`umap_alternative` と同じ Rust 1.93.1、CPython 3.13.12、uv 0.10.8 を使います。Windows では Visual Studio 2022 の C++ ツールと Windows SDK、Linux では C コンパイラとリンカーが必要です。

```sh
uv python install
uv sync --locked
cargo fmt --all -- --check
uv run --locked cargo clippy --workspace --all-targets --locked -- -D warnings
uv run --locked cargo test --workspace --locked
uv run --locked pytest
```

Rust の変更後は `uv sync --locked --reinstall-package hdbscan-alternative` で編集可能な Python 拡張を再ビルドできます。

## ビルド成果物

```sh
cargo build --release --locked -p hdbscan-core
cargo build --release --locked -p hdbscan-capi
uv run --locked maturin build --release --locked --interpreter python
```

- Rust からは `hdbscan-core` を依存クレートとして利用します。
- C ABI は `target/release/` に Windows の `hdbscan_capi.dll` とインポートライブラリ、Linux の `libhdbscan_capi.so` を生成します。宣言は [C ヘッダー](crates/hdbscan-capi/include/hdbscan_capi.h) にあります。
- Wheel は `target/wheels/` に生成されます。Python の配布名は `hdbscan-alternative`、インポート名は `hdbscan_alternative` です。

## クラスタリング

```python
from hdbscan_alternative import HDBSCAN

X = [[0.0], [0.0], [0.0], [2.0], [2.0], [2.0], [1.0]]
model = HDBSCAN(min_cluster_size=3, min_samples=2, gen_min_span_tree=True)
labels = model.fit_predict(X)
```

`fit(X)` は学習済み推定器を返し、`fit_predict(X)` はクラスタ番号の NumPy 配列を返します。ノイズは `-1` です。`probabilities_` と `cluster_persistence_` で所属強度と永続性を取得できます。`condensed_tree_` と `single_linkage_tree_` は `to_numpy()` で取得でき、`gen_min_span_tree=True` の場合は `minimum_spanning_tree_` も同様に取得できます。

`cluster_selection_method="leaf"` は凝縮木の末端にあるクラスタを選びます。凝縮木に根以外のクラスタがなければ全点がノイズです。EOM で `allow_single_cluster=True` を指定すると根も候補になり、根が選ばれた場合も根から早く離れた点はノイズとして残ります。[上流のパラメーター解説](https://hdbscan.readthedocs.io/en/latest/parameter_selection.html#leaf-clustering)

Rust では `FitOptions::with_cluster_selection` で設定を作り、入力形式に対応する `fit_dense_configured`、`fit_csr_configured` などに渡します。C では `hdbscan_capi_fit_configuration_t` を `hdbscan_capi_fit_dense_configured` または `hdbscan_capi_fit_csr_configured` に渡します。事前計算距離の場合は C の距離指定に `HDBSCAN_CAPI_METRIC_PRECOMPUTED` を使います。[C ヘッダー](crates/hdbscan-capi/include/hdbscan_capi.h)

`cluster_selection_epsilon` は距離の下限を指定し、近いクラスタを統合します。`cluster_selection_epsilon_max` と `max_cluster_size` は EOM の候補に対する距離とサイズの上限です。`cluster_selection_persistence` はクラスタ選択前に凝縮木の短命な枝を統合します。既定値は順に `0.0`、正の無限大、`0`、`0.0` で、サイズの `0` は上限なしを表します。[上流の API](https://hdbscan.readthedocs.io/en/latest/api.html#hdbscan)

Rust では `ClusterSelectionThresholds` を `FitConfiguration::with_cluster_selection_thresholds` に渡します。`FitConfiguration` を構造体リテラルで初期化する場合、通常のクラスタ割り当てには `strict_cluster_membership: false` を指定します。C では `hdbscan_capi_cluster_selection_thresholds_t` を `hdbscan_capi_fit_dense_with_thresholds` または `hdbscan_capi_fit_csr_with_thresholds` に渡します。[C ヘッダー](crates/hdbscan-capi/include/hdbscan_capi.h)

Python の初期設定は `algorithm="best"`、ユークリッド距離、`core_dist_n_jobs=4`、`approx_min_span_tree=True`、EOM です。特徴量の学習には、全列が有限の行が2行以上必要です。`min_samples=None` は `min_cluster_size` を使い、コア距離の計算では他の点の数を上限とします。Rust の `fit_dense_euclidean` は行優先の `&[f64]` を受け取り、すべての木を返します。C の `hdbscan_capi_fit_dense_euclidean` も行優先で、結果ハンドルからラベル、所属強度、永続性を借用できます。使用後は `hdbscan_capi_result_free` で解放します。[C ヘッダー](crates/hdbscan-capi/include/hdbscan_capi.h)

Python の関数版も利用できます。既定の Minkowski 距離（`p=2`）では有限の特徴量を渡します。関数は NumPy 配列を返し、`gen_min_span_tree=False` の場合は最後の要素が `None` になります。事前計算距離でも MST を取得できます。

```python
from hdbscan_alternative import hdbscan

labels, probabilities, persistence, condensed, linkage, mst = hdbscan(
    [[0.0], [1.0], [5.0], [6.0]], min_cluster_size=2, min_samples=1,
    gen_min_span_tree=True,
)
```

`HDBSCAN` と関数版で `match_reference_implementation=True` を指定できます。学習時に `min_cluster_size` を1増やし、`min_samples` を1減らします。`min_samples` の下限は1です。近似 MST を無効にし、選択された非ルートクラスタの生成時と同じ λ で離れた点をノイズとして扱います。[上流 API のオプション](https://hdbscan.readthedocs.io/en/latest/api.html#hdbscan)

`HDBSCAN(metric="euclidean")` には SciPy の疎行列・疎配列も渡せます。CSR の格納値から点間距離を計算し、同じ列の重複や格納順を保持します。Python では `float64` または整数の疎特徴量を使えます。`float32` の疎特徴量は `X.astype(np.float64)` で変換してから渡してください。Rust の `fit_csr_euclidean` と C の `hdbscan_capi_fit_csr_euclidean` には CSR の行オフセット、列番号、値、特徴量数を渡します。[C ヘッダー](crates/hdbscan-capi/include/hdbscan_capi.h)

密・疎の特徴量で `metric="manhattan"`（別名 `cityblock`、`l1`）と `metric="cosine"`（別名 `arccos`）も使えます。密な cosine 入力では `algorithm="generic"` を指定します。Manhattan は `float32` と `float64` を受け付け、CSR では内部バッファ上で重複列を合算します。cosine は `1 − cosine similarity` を使い、同一行の距離を `0`、ゼロベクトルと別の行の距離を `1` とします。`float32` の cosine 入力は `float64` に変換して渡してください。[Manhattan 距離](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.manhattan_distances.html) · [cosine 距離](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_distances.html)

Rust からは `fit_dense` と `fit_csr` に `FeatureMetric` を指定します。`fit_csr_manhattan_f32` は CSR の `float32` 差分を再現します。C では `hdbscan_capi_fit_dense_with_metric` と `hdbscan_capi_fit_csr_with_metric` に距離の定数を指定し、`double` の特徴量を渡します。[C ヘッダー](crates/hdbscan-capi/include/hdbscan_capi.h)

特徴量に `NaN`、`+inf`、`-inf` を含む行は除外され、ラベル `-1` と所属強度 `0.0` が元の行位置に返ります。凝縮木と単連結階層は元の行番号を使い、MST は学習に残った行の連番を使います。非有限値を含む CSR の重複列は内部コピーで合算してから有効行を判定します。Rust では `fit_dense_with_missing`、`fit_csr_with_missing`、`fit_csr_manhattan_f32_with_missing` を使います。C では `hdbscan_capi_fit_dense_with_missing` と `hdbscan_capi_fit_csr_with_missing` を使います。[C ヘッダー](crates/hdbscan-capi/include/hdbscan_capi.h)

Python の `metric="precomputed"`、Rust の `fit_dense_precomputed`、C の `hdbscan_capi_fit_dense_precomputed` には、サンプル間距離の正方行列を渡します。`+inf` は欠損距離を表します。格納された `NaN` は入力エラーになります。[事前計算距離の上流仕様](https://hdbscan.readthedocs.io/en/latest/basic_hdbscan.html#distance-matrices)

密な距離行列で、有限の相互到達可能距離で全点を結べない場合、Python の推定器と関数版は上流 Wheel 0.8.44 で観測した階層を返します。連結できなかった部分には `float64` の最大値を重みとする行が入ります。凝縮木の容量を超える場合は `ValueError` を返します。Python の `gen_min_span_tree=True` で、学習した推定器から `minimum_spanning_tree_` を参照すると警告とともに `None` が返ります。関数版からは MST の配列を取得できます。

Rust では `fit_dense_precomputed_bounded` で Python と同じ階層を取得できます。`fit_dense_precomputed` と C の `hdbscan_capi_fit_dense_precomputed` は、有限の相互到達可能距離で全点を結べない場合に非連結エラーを返します。

同じ `metric="precomputed"` で、SciPy の CSR・CSC・COO などの疎行列や疎配列も渡せます。Rust の `fit_csr_precomputed` と C の `hdbscan_capi_fit_csr_precomputed` は CSR の行オフセット、列番号、距離を受け付けます。疎行列の距離には有限値を使い、未格納の辺は欠損として扱います。正の `max_dist` を指定すると、近傍不足などで無限大になった格納辺の相互到達可能距離を指定値で置き換えます。既定値は `0.0` で、置き換えを無効にします。疎グラフの計算は CSR の格納要素を走査し、Kruskal 法で MST を作ります。[C ヘッダー](crates/hdbscan-capi/include/hdbscan_capi.h) · [SciPy の MST 仕様](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csgraph.minimum_spanning_tree.html)

## 学習経路の選択

密なユークリッド・Manhattan 特徴量では、Python の `algorithm="prims_kdtree"` と `algorithm="prims_balltree"` を指定できます。`float32` の特徴量も受け付けます。Rust は点間距離を直接計算して近傍の距離を求め、Prim 法で MST を作ります。Rust は `fit_dense_prims`、C は `hdbscan_capi_fit_dense_prims` を使います。[C ヘッダー](crates/hdbscan-capi/include/hdbscan_capi.h)

`algorithm="best"` は、密なユークリッド・Manhattan 特徴量で60列以下なら Borůvka の KD-tree、61列以上なら Prim を使います。疎な入力と事前計算距離では `generic` を使います。Python では `boruvka_kdtree`、`boruvka_balltree` を明示できます。Rust は `fit_dense_boruvka`、C は `hdbscan_capi_fit_dense_boruvka` を使います。[C ヘッダー](crates/hdbscan-capi/include/hdbscan_capi.h) · [上流の学習オプション](https://hdbscan.readthedocs.io/en/latest/api.html#hdbscan)

### 木探索で使える距離

密な特徴量では、KD-tree にユークリッド・Manhattan・Chebyshev・Minkowski を指定できます。Ball-tree では、さらに標準化ユークリッド、Mahalanobis、Hamming、Canberra、Bray–Curtis、Jaccard、Dice、Rogers–Tanimoto、Russell–Rao、Sokal–Michener、Sokal–Sneath、Haversine を使えます。Rust は `TreeDistance`、C は `hdbscan_capi_tree_distance_t` を使います。[C ヘッダー](crates/hdbscan-capi/include/hdbscan_capi.h)

公開版と同様に、Python の `HDBSCAN(metric="minkowski", p=3.5)` は `generic` で指定した指数を使い、木探索では既定の指数 `2.0` を使います。`RobustSingleLinkage(metric="minkowski", metric_params={"p": 3.5})` と Rust・C の距離指定では、木探索にも指数を渡せます。

## 学習後のスコアとクラスタリング

Python の `outlier_scores_` は凝縮木から各サンプルの GLOSH スコアを求めます。値が大きいほど外れ値の傾向が強くなります。初回アクセスで計算した NumPy 配列を保持し、同じ推定器を再学習した後も取得済みの配列を返します。離脱時の λ が有限で、所属した枝の最大 λ が無限大の点では `NaN` になります。[上流の外れ値検出](https://hdbscan.readthedocs.io/en/latest/outlier_detection.html)

`dbscan_clustering(cut_distance, min_cluster_size=5)` は単連結階層から DBSCAN* のラベルを求めます。結合距離が `cut_distance` 未満の辺を使い、最小サイズを下回る成分は `-1` になります。最小サイズを `1` にすると、学習時に除外された欠損行も単独のクラスタになります。[上流の公開 API](https://hdbscan.readthedocs.io/en/latest/api.html#hdbscan.hdbscan_.HDBSCAN.dbscan_clustering)

Rust では学習結果の `condensed_tree` を `outlier_scores` に、`single_linkage_tree` を `dbscan_clustering` に渡します。C ABI では学習結果と `n_samples` 要素の出力バッファを `hdbscan_capi_result_outlier_scores` または `hdbscan_capi_result_dbscan_clustering` に渡します。スコアとラベルは元の入力行順で返ります。[C ヘッダー](crates/hdbscan-capi/include/hdbscan_capi.h)

## 新しいデータの予測

```python
from hdbscan_alternative import HDBSCAN, approximate_predict, approximate_predict_scores

X = [[0.0], [0.1], [0.3], [1.0], [1.2], [10.0], [10.2], [10.6], [12.0], [18.0]]
model = HDBSCAN(min_cluster_size=2, min_samples=2, prediction_data=True).fit(X)
queries = [[0.05], [10.35]]
labels, strengths, neighbors = approximate_predict(model, queries, return_connecting_points=True)
outlier_scores = approximate_predict_scores(model, queries)
```

新しい点は学習済みの凝縮木に割り当てます。Python ではユークリッド、Manhattan、Minkowski、Chebyshev の密な有限特徴量を使えます。学習後に `generate_prediction_data()` で予測データを作り、`exemplars_` でクラスタの代表点を取得できます。凝縮木にクラスタ間の枝がない場合、公開版と同様に警告を出し、ラベル `-1`、所属強度 `0.0`、外れ値スコア `1` を返します。[上流の予測チュートリアル](https://hdbscan.readthedocs.io/en/latest/prediction_tutorial.html)

上流 Wheel 0.8.44 と同様に、`HDBSCAN(metric="minkowski", p=3.5)` の予測と枝検出は、近傍のキャッシュを既定の指数 `2.0` で作ります。

Rust は `PredictionModel::new` に学習データ、学習結果のラベルと凝縮木を渡します。Minkowski と Chebyshev には `PredictionModel::new_with_distance` を使います。C は `hdbscan_capi_prediction_create` でユークリッドまたは Manhattan の予測データを準備し、`hdbscan_capi_approximate_predict` と `hdbscan_capi_approximate_predict_scores` で結果を取得します。予測モデルは学習データと学習結果から独立して保持でき、使用後は `hdbscan_capi_prediction_free` で解放します。[C ヘッダー](crates/hdbscan-capi/include/hdbscan_capi.h)

## ソフトクラスタリング

```python
import numpy as np
from hdbscan_alternative import all_points_membership_vectors, membership_vector

new_memberships = membership_vector(model, np.asarray(queries))
training_memberships = all_points_membership_vectors(model)
```

各行が1点、各列が選択された1クラスタの所属確率です。列は凝縮木のクラスタ ID の昇順です。予測で対応する距離と、上の例で作成した予測データを使います。学習点と新規点は、それぞれの公開 API の計算手順で評価します。クラスタが選択されなかった場合、Python の `all_points_membership_vectors` は学習行数分の一次元のゼロ配列を返します。[上流の公開 API](https://hdbscan.readthedocs.io/en/latest/api.html#hdbscan.prediction.all_points_membership_vectors)

Rust では `PredictionModel` の同名メソッドが行優先の `MembershipMatrix` を返します。C では `hdbscan_capi_membership_shape` で列数を取得し、`hdbscan_capi_membership_vector` または `hdbscan_capi_all_points_membership_vectors` に行数×列数の `double` バッファを渡します。学習点の結果でクラスタが0件の場合、ネイティブ API の行列は0列です。[C ヘッダー](crates/hdbscan-capi/include/hdbscan_capi.h)

## クラスタリングの妥当性

```python
import numpy as np
from hdbscan_alternative import validity_index

points = np.array([[0.0], [1.0], [5.0], [6.0]])
labels = np.array([0, 0, 1, 1])
overall, per_cluster = validity_index(points, labels, per_cluster_scores=True)
```

`validity_index` は既存のクラスタラベルを使って DBCV を計算します。負のラベルはノイズとして扱い、全体スコアの重みには全行数を使います。クラスタ別スコアはラベルを添字とし、未使用ラベルの位置には `NaN` が入ります。Python では `metric="precomputed"` に元の特徴空間の次元数 `d` を指定します。公開版 0.8.44 と同じく、事前計算距離と `mst_raw_dist=True` の組み合わせは、クラスタ内の距離行列が C 連続でなければ `ValueError` を返します。[上流の公開 API](https://hdbscan.readthedocs.io/en/latest/api.html#hdbscan.validity.validity_index)

Rust は `validity::validity_index_dense` と `validity::validity_index_precomputed`、C は `hdbscan_capi_validity_index` を提供します。密な Euclidean・Manhattan・Cosine 特徴量と、行優先の事前計算距離を扱えます。ネイティブ API の事前計算距離は `mst_raw_dist` にも対応します。[C ヘッダー](crates/hdbscan-capi/include/hdbscan_capi.h)

### 学習済みモデルの相対妥当性

```python
from hdbscan_alternative import HDBSCAN

model = HDBSCAN(min_cluster_size=2, min_samples=1, gen_min_span_tree=True).fit(points)
score = model.relative_validity_
```

`relative_validity_` は学習時の相互到達可能距離の最小全域木を使う DBCV の近似値です。読み出した値は公開版と同じく再学習後も保持されます。再学習後の値を計算するには `model.relative_validity_ = None` を設定します。密な事前計算距離でも利用できます。CSR の事前計算距離で、木に両端がクラスタに属する辺がある場合は `ValueError` を返します。[上流の公開 API](https://hdbscan.readthedocs.io/en/latest/api.html#hdbscan.hdbscan_.HDBSCAN)

Rust は `validity::relative_validity` に学習結果のラベル、木の辺、`RelativeValidityTree` を渡します。C は `hdbscan_capi_result_relative_validity` に学習結果のハンドルを渡します。[C ヘッダー](crates/hdbscan-capi/include/hdbscan_capi.h)

## Robust Single Linkage

```python
from hdbscan_alternative import RobustSingleLinkage

model = RobustSingleLinkage(cut=2.5, k=2, gamma=2).fit(points)
labels = model.labels_
hierarchy = model.cluster_hierarchy_.to_numpy()
another_cut = model.cluster_hierarchy_.get_clusters(3.0, min_cluster_size=2)
```

`robust_single_linkage(points, cut=...)` でもラベルと階層を取得できます。密なユークリッド・Manhattan 特徴量では `best`、`generic`、KD-tree と Ball-tree の Prim・Borůvka を選べます。Python の `generic` は疎な特徴量、密な事前計算距離、および距離行列を計算できる追加の距離に対応します。Rust は `robust::robust_single_linkage_dense` と `robust::robust_single_linkage_precomputed`、C ABI は `hdbscan_capi_robust_single_linkage` を公開します。

## クラスタ内の枝検出と予測

```python
from hdbscan_alternative import HDBSCAN, BranchDetector, approximate_predict_branch

clusters = HDBSCAN(min_cluster_size=5, branch_detection_data=True, prediction_data=True).fit(points)
branches = BranchDetector(branch_detection_method="full").fit(clusters)
labels, strengths, cluster_ids, cluster_strengths, branch_ids, branch_strengths = (
    approximate_predict_branch(branches, new_points)
)
graph_points, graph_edges = branches.approximation_graph_.to_numpy()
```

Python では、ユークリッド、Manhattan、Minkowski、Chebyshev の密な特徴量から、`core` と `full` のグラフ、枝のラベル・所属強度・永続性、凝縮木、単連結階層を求めます。`detect_branches_in_clusters` では既存のクラスタラベルと所属強度を渡せます。Python の近似予測には `prediction_data=True` も指定します。グラフと枝の木は NumPy、pandas、NetworkX 形式で取得できます。

Rust では `branches::BranchDetectionData::new_with_distance` で距離を指定し、`detect` と `branches::BranchResult::classify` を使います。C ABI では、ユークリッドと Manhattan の枝検出・近似予測に `hdbscan_capi_detect_branches` と `hdbscan_capi_approximate_predict_branch` を使います。C の結果ビューは `hdbscan_capi_branch_result_free` で解放するまで利用できます。

## Python の描画と scikit-learn 連携

`HDBSCAN`、`RobustSingleLinkage`、`BranchDetector` は scikit-learn の `BaseEstimator` を継承します。`get_params`、`set_params`、`clone` が使えます。`HDBSCAN` と `RobustSingleLinkage` は `Pipeline` の最終段に置けます。

描画と表・グラフの出力に必要なライブラリは `uv sync --locked --extra plot` で導入できます。単連結階層、凝縮木、最小全域木、枝の近似グラフは `.plot()` で描画します。枝の近似グラフには `positions` を渡せます。省略時は `pygraphviz` を通じて Graphviz の `sfdp` で配置します。現行の PyGraphviz Wheel には Graphviz が同梱されています。検証では `uv run --no-sync --with pygraphviz==2.0.4 pytest tests/test_python_integration.py` を使います。[PyGraphviz の導入手順](https://pygraphviz.github.io/documentation/stable/install.html)

## 独立実装と数値互換性

実装で参照する資料は公開論文、自前の観測仕様、公開 API から得た比較データに限定します。検証では `reference/` の専用環境で `hdbscan` 0.8.44 の配布 Wheel を実行し、自作の入力、設定、実行環境、出力を JSON に記録します。ラベルの番号と出力順をそのまま比較し、浮動小数点出力はビット表現で照合します。一致を検証した経路から高速化します。[HDBSCAN 原論文](https://link.springer.com/chapter/10.1007/978-3-642-37456-2_14) · [参照 Wheel](https://pypi.org/project/hdbscan/0.8.44/)

同距離の辺の順序は、Windows と Linux の NumPy 2.5.5 の `X86_V3` 経路に合わせています。Linux の参照データ生成では AVX-512 向けの経路を無効にします。[NumPy の CPU 別の実行経路](https://numpy.org/doc/stable/reference/simd/build-options.html#runtime-dispatch)

```sh
uv run --project reference --locked python -I reference/generate_fixture.py --output hdbscan-reference.json
```

Iris、Wine、Breast Cancer Wisconsin、Digits は `reference/compare_public_datasets.py` で照合します。上流 Wheel を使った出力と Rust 版の出力について、配列の型・形状・バイト列のハッシュ、および例外と警告を CI で比較します。[scikit-learn のデータセット](https://scikit-learn.org/stable/datasets/toy_dataset.html)

## ライセンス

[MIT License](LICENSE)
