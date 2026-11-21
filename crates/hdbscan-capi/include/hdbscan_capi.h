#ifndef HDBSCAN_CAPI_H
#define HDBSCAN_CAPI_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Returns the library version as a NUL-terminated ASCII string.
 * The library owns the string. It remains valid until the library is unloaded. */
const char *hdbscan_capi_version(void);

typedef uint32_t hdbscan_capi_status_t;
enum {
    HDBSCAN_CAPI_STATUS_OK = 0,
    HDBSCAN_CAPI_STATUS_NULL_ARGUMENT = 1,
    HDBSCAN_CAPI_STATUS_INVALID_POINTER = 2,
    HDBSCAN_CAPI_STATUS_INVALID_LENGTH = 3,
    HDBSCAN_CAPI_STATUS_INVALID_SHAPE = 4,
    HDBSCAN_CAPI_STATUS_INVALID_OPTIONS = 5,
    HDBSCAN_CAPI_STATUS_NONFINITE_INPUT = 6,
    HDBSCAN_CAPI_STATUS_DISTANCE_OVERFLOW = 7,
    HDBSCAN_CAPI_STATUS_PANIC = 8,
    HDBSCAN_CAPI_STATUS_DISCONNECTED_DISTANCES = 9,
    HDBSCAN_CAPI_STATUS_INVALID_CSR = 10,
    HDBSCAN_CAPI_STATUS_INSUFFICIENT_SPARSE_NEIGHBORS = 11,
    HDBSCAN_CAPI_STATUS_INVALID_METRIC = 12
};

typedef uint32_t hdbscan_capi_metric_t;
enum {
    HDBSCAN_CAPI_METRIC_EUCLIDEAN = 0,
    HDBSCAN_CAPI_METRIC_MANHATTAN = 1,
    HDBSCAN_CAPI_METRIC_COSINE = 2,
    HDBSCAN_CAPI_METRIC_PRECOMPUTED = 3
};

typedef uint32_t hdbscan_capi_cluster_selection_method_t;
enum {
    HDBSCAN_CAPI_CLUSTER_SELECTION_EOM = 0,
    HDBSCAN_CAPI_CLUSTER_SELECTION_LEAF = 1
};

typedef struct hdbscan_capi_options {
    size_t min_cluster_size;
    size_t min_samples; /* 0 uses min_cluster_size. */
    double alpha;
} hdbscan_capi_options_t;

typedef struct hdbscan_capi_fit_configuration {
    hdbscan_capi_options_t options;
    hdbscan_capi_cluster_selection_method_t cluster_selection_method;
    uint32_t allow_single_cluster; /* 0 or 1. Allows the root in EOM and epsilon merging. */
    double max_dist; /* CSR precomputed fallback. Pass 0.0 to disable it. */
} hdbscan_capi_fit_configuration_t;

typedef struct hdbscan_capi_cluster_selection_thresholds {
    double epsilon; /* Minimum merge distance. Defaults to 0.0. */
    double epsilon_max; /* EOM maximum distance. Defaults to positive infinity. */
    double persistence; /* Minimum branch lifetime. Defaults to 0.0. */
    double max_cluster_size; /* EOM size limit. Nonpositive values disable it. */
} hdbscan_capi_cluster_selection_thresholds_t;

typedef struct hdbscan_capi_csr {
    const size_t *indptr; /* n_samples + 1 row offsets, starting at 0 and ending at nnz. */
    const size_t *indices; /* nnz zero-based column indices. */
    const double *data; /* nnz finite values. */
    size_t n_samples;
    size_t nnz;
} hdbscan_capi_csr_t;

enum {
    HDBSCAN_CAPI_BORUVKA_KDTREE = 0,
    HDBSCAN_CAPI_BORUVKA_BALLTREE = 1
};

typedef struct hdbscan_capi_boruvka_options {
    uint32_t tree;
    uint32_t approximate; /* 0 for exact, 1 for approximate. */
    size_t leaf_size;
    int64_t core_dist_n_jobs;
} hdbscan_capi_boruvka_options_t;

enum {
    HDBSCAN_CAPI_TREE_EUCLIDEAN = 0, HDBSCAN_CAPI_TREE_MANHATTAN = 1,
    HDBSCAN_CAPI_TREE_CHEBYSHEV = 2, HDBSCAN_CAPI_TREE_MINKOWSKI = 3,
    HDBSCAN_CAPI_TREE_SEUCLIDEAN = 4, HDBSCAN_CAPI_TREE_MAHALANOBIS = 5,
    HDBSCAN_CAPI_TREE_HAMMING = 6, HDBSCAN_CAPI_TREE_CANBERRA = 7,
    HDBSCAN_CAPI_TREE_BRAYCURTIS = 8, HDBSCAN_CAPI_TREE_JACCARD = 9,
    HDBSCAN_CAPI_TREE_DICE = 10, HDBSCAN_CAPI_TREE_ROGERSTANIMOTO = 11,
    HDBSCAN_CAPI_TREE_RUSSELLRAO = 12, HDBSCAN_CAPI_TREE_SOKALMICHENER = 13,
    HDBSCAN_CAPI_TREE_SOKALSNEATH = 14, HDBSCAN_CAPI_TREE_HAVERSINE = 15
};

typedef struct hdbscan_capi_tree_distance {
    uint32_t kind;
    double p;                /* Minkowski exponent. */
    const double *parameters; /* Optional Minkowski weights, variances or row-major VI. */
    size_t parameter_count;
} hdbscan_capi_tree_distance_t;

typedef struct hdbscan_capi_result hdbscan_capi_result_t;
typedef struct hdbscan_capi_prediction_model hdbscan_capi_prediction_model_t;
typedef struct hdbscan_capi_branch_result hdbscan_capi_branch_result_t;

typedef struct hdbscan_capi_linkage_row {
    size_t left;
    size_t right;
    double distance;
    size_t size;
} hdbscan_capi_linkage_row_t;

typedef struct hdbscan_capi_condensed_edge {
    size_t parent;
    size_t child;
    double lambda;
    size_t child_size;
} hdbscan_capi_condensed_edge_t;

enum {
    HDBSCAN_CAPI_ROBUST_BEST = 0, HDBSCAN_CAPI_ROBUST_GENERIC = 1,
    HDBSCAN_CAPI_ROBUST_PRIM = 2, HDBSCAN_CAPI_ROBUST_BORUVKA_KDTREE = 3,
    HDBSCAN_CAPI_ROBUST_BORUVKA_BALLTREE = 4
};

typedef struct hdbscan_capi_robust_options {
    double cut;
    size_t k;
    double alpha;
    int64_t gamma;
    uint32_t algorithm;
    size_t leaf_size;
    int64_t core_dist_n_jobs;
} hdbscan_capi_robust_options_t;

enum { HDBSCAN_CAPI_BRANCH_FULL = 0, HDBSCAN_CAPI_BRANCH_CORE = 1 };

typedef struct hdbscan_capi_branch_options {
    size_t min_samples;
    size_t min_cluster_size;
    uint32_t method;
    uint32_t label_sides_as_branches;
    hdbscan_capi_cluster_selection_method_t cluster_selection_method;
    uint32_t allow_single_cluster;
} hdbscan_capi_branch_options_t;

typedef struct hdbscan_capi_branch_result_view {
    size_t n_samples;
    size_t n_clusters;
    const int64_t *labels;
    const double *probabilities;
    const int64_t *cluster_labels;
    const double *cluster_probabilities;
    const int64_t *branch_labels;
    const double *branch_probabilities;
    const double *centralities;
} hdbscan_capi_branch_result_view_t;

typedef struct hdbscan_capi_branch_edge {
    size_t source;
    size_t target;
    double centrality;
    double reachability;
} hdbscan_capi_branch_edge_t;

typedef struct hdbscan_capi_branch_cluster_view {
    const size_t *points;
    size_t n_points;
    const hdbscan_capi_branch_edge_t *graph;
    size_t n_edges;
    const hdbscan_capi_linkage_row_t *linkage;
    size_t n_merges;
    const hdbscan_capi_condensed_edge_t *condensed;
    size_t n_condensed;
    const double *persistences;
    size_t n_persistences;
} hdbscan_capi_branch_cluster_view_t;

typedef struct hdbscan_capi_branch_prediction {
    int64_t label;
    double probability;
    int64_t cluster_label;
    double cluster_probability;
    int64_t branch_label;
    double branch_probability;
} hdbscan_capi_branch_prediction_t;

typedef struct hdbscan_capi_prediction_row {
    int64_t label;
    double probability;
    int64_t connecting_point;
} hdbscan_capi_prediction_row_t;

typedef struct hdbscan_capi_membership_shape {
    size_t n_samples;
    size_t n_clusters;
} hdbscan_capi_membership_shape_t;

typedef struct hdbscan_capi_result_view {
    const int64_t *labels; /* n_samples entries. Noise has label -1. */
    const double *probabilities; /* n_samples entries. */
    size_t n_samples;
    const double *cluster_persistence; /* n_clusters entries, or NULL for zero clusters. */
    size_t n_clusters;
} hdbscan_capi_result_view_t;

/* Fits dense Euclidean HDBSCAN with exact MST and EOM selection. The input is
 * row-major, with data_len = n_samples * n_features. NULL options select
 * min_cluster_size = 5, min_samples = 0, alpha = 1.0. The caller owns the new
 * result on success. A valid out pointer is set to NULL on failure. Release
 * any previous result before reusing its output slot. */
hdbscan_capi_status_t hdbscan_capi_fit_dense_euclidean(
    const double *data, size_t data_len, size_t n_samples, size_t n_features,
    const hdbscan_capi_options_t *options, hdbscan_capi_result_t **out);

/* Fits dense features with the selected metric. Ownership follows dense fit. */
hdbscan_capi_status_t hdbscan_capi_fit_dense_with_metric(
    const double *data, size_t data_len, size_t n_samples, size_t n_features,
    hdbscan_capi_metric_t metric, const hdbscan_capi_options_t *options,
    hdbscan_capi_result_t **out);

/* Fits a square distance matrix with data_len = n_samples * n_samples.
 * Positive infinity marks missing distances. All samples must be connected
 * by finite mutual reachability distances. Ownership follows the fit above. */
hdbscan_capi_status_t hdbscan_capi_fit_dense_precomputed(
    const double *data, size_t data_len, size_t n_samples,
    const hdbscan_capi_options_t *options, hdbscan_capi_result_t **out);

/* Fits a square CSR distance graph. An unstored entry is an absent edge.
 * A positive max_dist supplies a fallback weight for a stored edge whose
 * mutual reachability distance overflows or has an insufficient-neighbor
 * endpoint. Pass 0.0 to disable the fallback. Ownership follows dense fit. */
hdbscan_capi_status_t hdbscan_capi_fit_csr_precomputed(
    const hdbscan_capi_csr_t *matrix, const hdbscan_capi_options_t *options,
    double max_dist, hdbscan_capi_result_t **out);

/* Fits Euclidean HDBSCAN to n_samples rows of n_features CSR features.
 * The CSR values retain their stored order and duplicates. Ownership follows
 * dense fit. */
hdbscan_capi_status_t hdbscan_capi_fit_csr_euclidean(
    const hdbscan_capi_csr_t *matrix, size_t n_features,
    const hdbscan_capi_options_t *options, hdbscan_capi_result_t **out);

/* Fits CSR features with the selected metric. Ownership follows dense fit. */
hdbscan_capi_status_t hdbscan_capi_fit_csr_with_metric(
    const hdbscan_capi_csr_t *matrix, size_t n_features, hdbscan_capi_metric_t metric,
    const hdbscan_capi_options_t *options, hdbscan_capi_result_t **out);

/* Fit with cluster selection options. Both functions accept precomputed
 * distances when n_features equals n_samples. NULL configuration uses default
 * fit options and EOM with the root excluded. Ownership follows dense fit. */
hdbscan_capi_status_t hdbscan_capi_fit_dense_configured(
    const double *data, size_t data_len, size_t n_samples, size_t n_features,
    hdbscan_capi_metric_t metric, const hdbscan_capi_fit_configuration_t *configuration,
    hdbscan_capi_result_t **out);
hdbscan_capi_status_t hdbscan_capi_fit_csr_configured(
    const hdbscan_capi_csr_t *matrix, size_t n_features, hdbscan_capi_metric_t metric,
    const hdbscan_capi_fit_configuration_t *configuration, hdbscan_capi_result_t **out);

/* Fit with additional thresholds. NULL thresholds uses the defaults above. */
hdbscan_capi_status_t hdbscan_capi_fit_dense_with_thresholds(
    const double *data, size_t data_len, size_t n_samples, size_t n_features,
    hdbscan_capi_metric_t metric, const hdbscan_capi_fit_configuration_t *configuration,
    const hdbscan_capi_cluster_selection_thresholds_t *thresholds, hdbscan_capi_result_t **out);
hdbscan_capi_status_t hdbscan_capi_fit_csr_with_thresholds(
    const hdbscan_capi_csr_t *matrix, size_t n_features, hdbscan_capi_metric_t metric,
    const hdbscan_capi_fit_configuration_t *configuration,
    const hdbscan_capi_cluster_selection_thresholds_t *thresholds, hdbscan_capi_result_t **out);

/* Fit features with NaN or infinity in any column treated as a noise row.
 * At least two finite rows are required. The result is in original row order.
 * These functions accept Euclidean, Manhattan, and cosine features. */
hdbscan_capi_status_t hdbscan_capi_fit_dense_with_missing(
    const double *data, size_t data_len, size_t n_samples, size_t n_features,
    hdbscan_capi_metric_t metric, const hdbscan_capi_fit_configuration_t *configuration,
    const hdbscan_capi_cluster_selection_thresholds_t *thresholds, hdbscan_capi_result_t **out);
hdbscan_capi_status_t hdbscan_capi_fit_csr_with_missing(
    const hdbscan_capi_csr_t *matrix, size_t n_features, hdbscan_capi_metric_t metric,
    const hdbscan_capi_fit_configuration_t *configuration,
    const hdbscan_capi_cluster_selection_thresholds_t *thresholds, hdbscan_capi_result_t **out);

/* Fit dense Euclidean or Manhattan features using the exact Prim vector path.
 * Rows containing NaN or infinity are reported as noise. */
hdbscan_capi_status_t hdbscan_capi_fit_dense_prims(
    const double *data, size_t data_len, size_t n_samples, size_t n_features,
    hdbscan_capi_metric_t metric, const hdbscan_capi_fit_configuration_t *configuration,
    const hdbscan_capi_cluster_selection_thresholds_t *thresholds, hdbscan_capi_result_t **out);

/* Fit dense Euclidean or Manhattan features using Boruvka's spatial search.
 * NULL search selects KD-tree, an approximate MST, and leaf_size 40. */
hdbscan_capi_status_t hdbscan_capi_fit_dense_boruvka(
    const double *data, size_t data_len, size_t n_samples, size_t n_features,
    hdbscan_capi_metric_t metric, const hdbscan_capi_fit_configuration_t *configuration,
    const hdbscan_capi_cluster_selection_thresholds_t *thresholds,
    const hdbscan_capi_boruvka_options_t *search, hdbscan_capi_result_t **out);

/* Pass NULL search for Prim, or configure a KD-tree or Ball-tree for Boruvka. */
hdbscan_capi_status_t hdbscan_capi_fit_dense_tree(
    const double *data, size_t data_len, size_t n_samples, size_t n_features,
    const hdbscan_capi_tree_distance_t *distance, const hdbscan_capi_fit_configuration_t *configuration,
    const hdbscan_capi_cluster_selection_thresholds_t *thresholds,
    const hdbscan_capi_boruvka_options_t *search, hdbscan_capi_result_t **out);

/* The result owns all data in the view. The pointers may be used until the
 * caller frees the result. On failure this function leaves out untouched. */
hdbscan_capi_status_t hdbscan_capi_result_view(
    const hdbscan_capi_result_t *result, hdbscan_capi_result_view_t *out);

/* Write one value per original sample to a caller-owned buffer.
 * Returns INVALID_LENGTH if out_len differs from the sample count.
 * The output buffer is untouched on error.
 * DBSCAN* includes only merges whose distance is strictly below cut_distance. */
hdbscan_capi_status_t hdbscan_capi_result_outlier_scores(
    const hdbscan_capi_result_t *result, double *out, size_t out_len);

/* Compute the fitted tree's relative validity and write it to out on success.
 * CSR precomputed results whose tree connects labeled points return INVALID_SHAPE. */
hdbscan_capi_status_t hdbscan_capi_result_relative_validity(
    const hdbscan_capi_result_t *result, double *out);
hdbscan_capi_status_t hdbscan_capi_result_dbscan_clustering(
    const hdbscan_capi_result_t *result, double cut_distance, int64_t min_cluster_size,
    int64_t *out, size_t out_len);

/* Copies the dense row-major training matrix and the data needed from its fit.
 * The caller may release both after this call. Supported metrics are
 * EUCLIDEAN and MANHATTAN. Pass the effective min_samples (min_cluster_size
 * when min_samples was omitted). On failure *out is NULL if available. */
hdbscan_capi_status_t hdbscan_capi_prediction_create(
    const hdbscan_capi_result_t *result, const double *data, size_t data_len,
    size_t n_features, hdbscan_capi_metric_t metric, size_t min_samples,
    hdbscan_capi_prediction_model_t **out);

/* Predict on out_len row-major samples, with the same feature count as the
 * training data. For an empty batch neither buffer is needed. Output is
 * untouched on failure. Connecting indices refer to the training row order. */
hdbscan_capi_status_t hdbscan_capi_approximate_predict(
    const hdbscan_capi_prediction_model_t *model, const double *data, size_t data_len,
    hdbscan_capi_prediction_row_t *out, size_t out_len);
hdbscan_capi_status_t hdbscan_capi_approximate_predict_scores(
    const hdbscan_capi_prediction_model_t *model, const double *data, size_t data_len,
    double *out, size_t out_len);

/* Return the number of training rows and selected clusters. */
hdbscan_capi_status_t hdbscan_capi_membership_shape(
    const hdbscan_capi_prediction_model_t *model, hdbscan_capi_membership_shape_t *out);

/* Write one row per query or training sample and one column per selected
 * cluster in increasing cluster ID order. out_len counts doubles, and
 * the output is unchanged on failure. An empty training matrix with zero
 * selected clusters requires no output buffer. */
hdbscan_capi_status_t hdbscan_capi_membership_vector(
    const hdbscan_capi_prediction_model_t *model, const double *data, size_t data_len,
    double *out, size_t out_len);
hdbscan_capi_status_t hdbscan_capi_all_points_membership_vectors(
    const hdbscan_capi_prediction_model_t *model, double *out, size_t out_len);

/* Evaluate DBCV on row-major data with one label per row. Negative labels
 * count as noise. For precomputed distances, n_features must equal n_samples
 * and dimension gives the original feature dimension. Other metrics use
 * n_features for the dimension. Pass 0 or 1 for mst_raw_dist.
 * overall is required. Pass NULL and 0 to omit the per-label output, or
 * provide a buffer with max(labels) + 1 slots. Unused label slots contain NaN.
 * Outputs are written on success. Buffers must not overlap. */
hdbscan_capi_status_t hdbscan_capi_validity_index(
    const double *data, size_t data_len, const int64_t *labels, size_t n_samples,
    size_t n_features, hdbscan_capi_metric_t metric, double dimension,
    uint32_t mst_raw_dist, double *overall, double *per_cluster, size_t per_cluster_len);

void hdbscan_capi_prediction_free(hdbscan_capi_prediction_model_t *model);

/* Fit robust single linkage to dense row-major features or distances.
 * On success write n_samples labels and n_samples-1 hierarchy rows.
 * NULL options use the RobustSingleLinkage defaults. */
hdbscan_capi_status_t hdbscan_capi_robust_single_linkage(
    const double *data, size_t data_len, size_t n_samples, size_t n_features,
    hdbscan_capi_metric_t metric, const hdbscan_capi_robust_options_t *options,
    int64_t *labels, size_t labels_len, hdbscan_capi_linkage_row_t *hierarchy, size_t hierarchy_len);

hdbscan_capi_status_t hdbscan_capi_robust_single_linkage_tree(
    const double *data, size_t data_len, size_t n_samples, size_t n_features,
    const hdbscan_capi_tree_distance_t *distance, const hdbscan_capi_robust_options_t *options,
    int64_t *labels, size_t labels_len, hdbscan_capi_linkage_row_t *hierarchy, size_t hierarchy_len);

/* Detect branches in a fit of the supplied dense Euclidean or Manhattan
 * features. The branch result owns its outputs. Pass the effective HDBSCAN
 * min_samples in options. NULL thresholds use the default thresholds. */
hdbscan_capi_status_t hdbscan_capi_detect_branches(
    const hdbscan_capi_result_t *result, const double *data, size_t data_len,
    size_t n_features, hdbscan_capi_metric_t metric, const hdbscan_capi_branch_options_t *options,
    const hdbscan_capi_cluster_selection_thresholds_t *thresholds, hdbscan_capi_branch_result_t **out);

/* The views borrow arrays until the branch result is freed. */
hdbscan_capi_status_t hdbscan_capi_branch_result_view(
    const hdbscan_capi_branch_result_t *result, hdbscan_capi_branch_result_view_t *out);
hdbscan_capi_status_t hdbscan_capi_branch_cluster_view(
    const hdbscan_capi_branch_result_t *result, size_t cluster, hdbscan_capi_branch_cluster_view_t *out);
/* Both handles must come from the same training fit. Write one prediction
 * per row of dense query data, using the fit's feature count. */
hdbscan_capi_status_t hdbscan_capi_approximate_predict_branch(
    const hdbscan_capi_branch_result_t *branches, const hdbscan_capi_prediction_model_t *prediction,
    const double *data, size_t data_len, hdbscan_capi_branch_prediction_t *out, size_t out_len);
void hdbscan_capi_branch_result_free(hdbscan_capi_branch_result_t *result);

/* Release the result and its views. Passing NULL has no effect. */
void hdbscan_capi_result_free(hdbscan_capi_result_t *result);

#ifdef __cplusplus
}
#endif

#endif /* HDBSCAN_CAPI_H */
