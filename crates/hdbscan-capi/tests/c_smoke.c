#include <float.h>
#include <math.h>
#include <stddef.h>

#include "hdbscan_capi.h"

static int check_branch_detection(void) {
    const double train[] = {0.0, 1.0, 3.0, 9.0, 11.0, 13.0, 48.0};
    const double queries[] = {1.0, 11.0, 48.0};
    const hdbscan_capi_options_t fit_options = {2, 1, 1.0};
    const hdbscan_capi_branch_options_t branch_options = {
        1, 2, HDBSCAN_CAPI_BRANCH_FULL, 0, HDBSCAN_CAPI_CLUSTER_SELECTION_EOM, 0
    };
    hdbscan_capi_result_t *fit = NULL;
    hdbscan_capi_branch_result_t *branches = NULL;
    hdbscan_capi_prediction_model_t *prediction = NULL;
    hdbscan_capi_branch_result_view_t view;
    hdbscan_capi_branch_cluster_view_t group;
    hdbscan_capi_branch_prediction_t predicted[3];
    int ok = hdbscan_capi_fit_dense_euclidean(train, 7, 7, 1, &fit_options, &fit) == HDBSCAN_CAPI_STATUS_OK
        && hdbscan_capi_detect_branches(fit, train, 7, 1, HDBSCAN_CAPI_METRIC_EUCLIDEAN,
            &branch_options, NULL, &branches) == HDBSCAN_CAPI_STATUS_OK
        && hdbscan_capi_branch_result_view(branches, &view) == HDBSCAN_CAPI_STATUS_OK
        && view.n_samples == 7 && view.n_clusters == 2
        && view.labels[0] == 0 && view.labels[3] == 1 && view.labels[6] == -1
        && view.probabilities[2] == 0.5 && isinf(view.centralities[1])
        && hdbscan_capi_branch_cluster_view(branches, 0, &group) == HDBSCAN_CAPI_STATUS_OK
        && group.n_points == 3 && group.n_edges == 2 && group.n_merges == 2
        && group.graph[0].source == 0 && group.graph[0].target == 1
        && isinf(group.graph[0].centrality) && group.graph[0].reachability == 1.0
        && hdbscan_capi_prediction_create(fit, train, 7, 1, HDBSCAN_CAPI_METRIC_EUCLIDEAN,
            1, &prediction) == HDBSCAN_CAPI_STATUS_OK
        && hdbscan_capi_approximate_predict_branch(branches, prediction, queries, 3, predicted, 3)
            == HDBSCAN_CAPI_STATUS_OK
        && predicted[0].label == 0 && predicted[1].label == 1 && predicted[2].label == -1
        && predicted[0].probability == 1.0 && predicted[2].probability == 0.0
        && predicted[0].branch_label == 0 && predicted[2].branch_probability == 1.0;
    hdbscan_capi_prediction_free(prediction);
    hdbscan_capi_branch_result_free(branches);
    hdbscan_capi_result_free(fit);
    return ok;
}

int main(void) {
    const char *version = hdbscan_capi_version();
    if (version == NULL || version[0] == '\0') {
        return 1;
    }

    const double x[] = {0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 1.0};
    const hdbscan_capi_options_t options = {3, 2, 1.0};
    hdbscan_capi_result_t *result = NULL;
    if (hdbscan_capi_fit_dense_euclidean(x, 6, 7, 1, &options, &result)
            != HDBSCAN_CAPI_STATUS_INVALID_LENGTH || result != NULL) {
        return 2;
    }

    if (hdbscan_capi_fit_dense_euclidean(x, 7, 7, 1, &options, &result) != HDBSCAN_CAPI_STATUS_OK) {
        return 3;
    }
    hdbscan_capi_result_view_t view;
    if (hdbscan_capi_result_view(result, &view) != HDBSCAN_CAPI_STATUS_OK
            || view.n_samples != 7 || view.n_clusters != 2) {
        hdbscan_capi_result_free(result);
        return 4;
    }

    const int64_t expected[] = {0, 0, 0, 1, 1, 1, 1};
    int exit_code = 0;
    for (size_t i = 0; i < view.n_samples; ++i) {
        if (view.labels[i] != expected[i] || view.probabilities[i] != (i == 6 ? 0.0 : 1.0)) {
            exit_code = 5;
        }
    }
    if (view.cluster_persistence[0] != 1.0 || view.cluster_persistence[1] != 1.0) {
        exit_code = 6;
    }
    hdbscan_capi_result_free(result);
    hdbscan_capi_result_free(NULL);
    if (exit_code != 0) {
        return exit_code;
    }

    double distances[] = {
        0.0, 1.0, INFINITY, INFINITY,
        1.0, 0.0, 5.0, INFINITY,
        INFINITY, 5.0, 0.0, 1.0,
        INFINITY, INFINITY, 1.0, 0.0
    };
    const hdbscan_capi_options_t matrix_options = {2, 1, 1.0};
    result = NULL;
    if (hdbscan_capi_fit_dense_precomputed(distances, 16, 4, &matrix_options, &result)
            != HDBSCAN_CAPI_STATUS_OK) {
        return 7;
    }
    if (hdbscan_capi_result_view(result, &view) != HDBSCAN_CAPI_STATUS_OK
            || view.n_samples != 4 || view.n_clusters != 2
            || view.labels[0] != 0 || view.labels[1] != 0
            || view.labels[2] != 1 || view.labels[3] != 1) {
        hdbscan_capi_result_free(result);
        return 8;
    }
    double relative_validity = -1.0;
    if (hdbscan_capi_result_relative_validity(NULL, &relative_validity)
            != HDBSCAN_CAPI_STATUS_NULL_ARGUMENT || relative_validity != -1.0
            || hdbscan_capi_result_relative_validity(result, &relative_validity)
            != HDBSCAN_CAPI_STATUS_OK || relative_validity != 0.8) {
        hdbscan_capi_result_free(result);
        return 60;
    }
    hdbscan_capi_result_free(result);

    distances[6] = INFINITY;
    distances[9] = INFINITY;
    result = NULL;
    if (hdbscan_capi_fit_dense_precomputed(distances, 16, 4, &matrix_options, &result)
            != HDBSCAN_CAPI_STATUS_DISCONNECTED_DISTANCES || result != NULL) {
        hdbscan_capi_result_free(result);
        return 9;
    }

    const size_t indptr[] = {0, 1, 3, 5, 6};
    const size_t columns[] = {1, 0, 2, 1, 3, 2};
    const double values[] = {1.0, 1.0, 5.0, 5.0, 1.0, 1.0};
    hdbscan_capi_csr_t csr = {indptr, columns, values, 4, 6};
    const hdbscan_capi_options_t sparse_options = {2, 2, 1.0};
    if (hdbscan_capi_fit_csr_precomputed(&csr, &sparse_options, 0.0, &result)
            != HDBSCAN_CAPI_STATUS_INSUFFICIENT_SPARSE_NEIGHBORS || result != NULL) {
        hdbscan_capi_result_free(result);
        return 10;
    }
    if (hdbscan_capi_fit_csr_precomputed(&csr, &sparse_options, 0.3, &result)
            != HDBSCAN_CAPI_STATUS_OK) {
        return 11;
    }
    if (hdbscan_capi_result_view(result, &view) != HDBSCAN_CAPI_STATUS_OK
            || view.n_samples != 4 || view.n_clusters != 2
            || view.labels[0] != 0 || view.labels[1] != 0
            || view.labels[2] != 1 || view.labels[3] != 1) {
        hdbscan_capi_result_free(result);
        return 12;
    }
    if (hdbscan_capi_result_relative_validity(result, &relative_validity)
            != HDBSCAN_CAPI_STATUS_INVALID_SHAPE || relative_validity != 0.8) {
        hdbscan_capi_result_free(result);
        return 61;
    }
    hdbscan_capi_result_free(result);
    result = NULL;

    const size_t bad_offsets[] = {0, 2, 1, 5, 6};
    csr.indptr = bad_offsets;
    if (hdbscan_capi_fit_csr_precomputed(&csr, &sparse_options, 0.3, &result)
            != HDBSCAN_CAPI_STATUS_INVALID_CSR || result != NULL) {
        hdbscan_capi_result_free(result);
        return 13;
    }

    const size_t feature_offsets[] = {0, 0, 0, 0, 1, 2, 3, 4};
    const size_t feature_columns[] = {7, 7, 7, 7};
    const double feature_data[] = {2.0, 2.0, 2.0, 1.0};
    const hdbscan_capi_csr_t features = {feature_offsets, feature_columns, feature_data, 7, 4};
    if (hdbscan_capi_fit_csr_euclidean(&features, 7, &options, &result)
            != HDBSCAN_CAPI_STATUS_INVALID_CSR || result != NULL) {
        hdbscan_capi_result_free(result);
        return 14;
    }
    if (hdbscan_capi_fit_csr_euclidean(&features, 8, &options, &result) != HDBSCAN_CAPI_STATUS_OK) {
        return 15;
    }
    if (hdbscan_capi_result_view(result, &view) != HDBSCAN_CAPI_STATUS_OK
            || view.n_samples != 7 || view.n_clusters != 2) {
        hdbscan_capi_result_free(result);
        return 16;
    }
    for (size_t i = 0; i < view.n_samples; ++i) {
        if (view.labels[i] != expected[i] || view.probabilities[i] != (i == 6 ? 0.0 : 1.0)) {
            hdbscan_capi_result_free(result);
            return 17;
        }
    }
    hdbscan_capi_result_free(result);
    result = NULL;

    const double signed_features[] = {1, 0, 2, 0, 5, 0, -1, 0, -2, 0, -5, 0};
    if (hdbscan_capi_fit_dense_with_metric(signed_features, 12, 6, 2, 999u, &options, &result)
            != HDBSCAN_CAPI_STATUS_INVALID_METRIC || result != NULL) {
        hdbscan_capi_result_free(result);
        return 18;
    }
    if (hdbscan_capi_fit_dense_with_metric(signed_features, 12, 6, 2,
            HDBSCAN_CAPI_METRIC_COSINE, &options, &result) != HDBSCAN_CAPI_STATUS_OK) {
        return 19;
    }
    if (hdbscan_capi_result_view(result, &view) != HDBSCAN_CAPI_STATUS_OK
            || view.n_samples != 6 || view.n_clusters != 2 || view.labels[0] == view.labels[3]) {
        hdbscan_capi_result_free(result);
        return 20;
    }
    for (size_t i = 0; i < view.n_samples; ++i) {
        if (view.labels[i] != view.labels[(i / 3) * 3] || view.probabilities[i] != 1.0) {
            hdbscan_capi_result_free(result);
            return 21;
        }
    }
    hdbscan_capi_result_free(result);
    result = NULL;

    if (hdbscan_capi_fit_csr_with_metric(&features, 8,
            HDBSCAN_CAPI_METRIC_MANHATTAN, &options, &result) != HDBSCAN_CAPI_STATUS_OK) {
        return 22;
    }
    if (hdbscan_capi_result_view(result, &view) != HDBSCAN_CAPI_STATUS_OK
            || view.n_samples != 7 || view.n_clusters != 2
            || view.labels[0] == view.labels[4] || view.probabilities[6] != 0.0) {
        hdbscan_capi_result_free(result);
        return 23;
    }
    hdbscan_capi_result_free(result);
    result = NULL;

    const double root_features[] = {0.0, 1.0, 5.0};
    hdbscan_capi_fit_configuration_t configuration = {
        {2, 1, 1.0}, HDBSCAN_CAPI_CLUSTER_SELECTION_EOM, 1, 0.0
    };
    if (hdbscan_capi_fit_dense_configured(root_features, 3, 3, 1,
            HDBSCAN_CAPI_METRIC_EUCLIDEAN, &configuration, &result) != HDBSCAN_CAPI_STATUS_OK) {
        return 24;
    }
    if (hdbscan_capi_result_view(result, &view) != HDBSCAN_CAPI_STATUS_OK
            || view.n_samples != 3 || view.n_clusters != 1
            || view.labels[0] != 0 || view.labels[1] != 0 || view.labels[2] != -1
            || view.probabilities[0] != 1.0 || view.probabilities[2] != 0.0
            || view.cluster_persistence[0] != 1.125) {
        hdbscan_capi_result_free(result);
        return 25;
    }
    hdbscan_capi_result_free(result);
    result = NULL;

    configuration.cluster_selection_method = 99;
    if (hdbscan_capi_fit_dense_configured(root_features, 3, 3, 1,
            HDBSCAN_CAPI_METRIC_EUCLIDEAN, &configuration, &result)
            != HDBSCAN_CAPI_STATUS_INVALID_OPTIONS || result != NULL) {
        hdbscan_capi_result_free(result);
        return 26;
    }

    configuration.cluster_selection_method = HDBSCAN_CAPI_CLUSTER_SELECTION_LEAF;
    configuration.options = sparse_options;
    configuration.max_dist = 0.3;
    csr.indptr = indptr;
    if (hdbscan_capi_fit_csr_configured(&csr, 4,
            HDBSCAN_CAPI_METRIC_PRECOMPUTED, &configuration, &result) != HDBSCAN_CAPI_STATUS_OK) {
        return 27;
    }
    if (hdbscan_capi_result_view(result, &view) != HDBSCAN_CAPI_STATUS_OK
            || view.n_samples != 4 || view.n_clusters != 2
            || view.labels[0] != 0 || view.labels[1] != 0
            || view.labels[2] != 1 || view.labels[3] != 1) {
        hdbscan_capi_result_free(result);
        return 28;
    }
    hdbscan_capi_result_free(result);
    result = NULL;

    configuration.cluster_selection_method = HDBSCAN_CAPI_CLUSTER_SELECTION_EOM;
    configuration.options = (hdbscan_capi_options_t){2, 1, 1.0};
    hdbscan_capi_cluster_selection_thresholds_t thresholds = {4.0, INFINITY, 0.0, 0.0};
    if (hdbscan_capi_fit_dense_with_thresholds(root_features, 3, 3, 1,
            HDBSCAN_CAPI_METRIC_EUCLIDEAN, &configuration, &thresholds, &result)
            != HDBSCAN_CAPI_STATUS_OK) {
        return 29;
    }
    if (hdbscan_capi_result_view(result, &view) != HDBSCAN_CAPI_STATUS_OK
            || view.n_clusters != 1 || view.labels[2] != 0
            || view.probabilities[2] != 0.25 || view.cluster_persistence[0] != 0.75) {
        hdbscan_capi_result_free(result);
        return 30;
    }
    hdbscan_capi_result_free(result);
    result = NULL;

    thresholds.epsilon_max = 3.0;
    if (hdbscan_capi_fit_dense_with_thresholds(root_features, 3, 3, 1,
            HDBSCAN_CAPI_METRIC_EUCLIDEAN, &configuration, &thresholds, &result)
            != HDBSCAN_CAPI_STATUS_INVALID_OPTIONS || result != NULL) {
        hdbscan_capi_result_free(result);
        return 31;
    }

    configuration.cluster_selection_method = HDBSCAN_CAPI_CLUSTER_SELECTION_LEAF;
    thresholds = (hdbscan_capi_cluster_selection_thresholds_t){6.0, INFINITY, 0.0, 0.0};
    if (hdbscan_capi_fit_csr_with_thresholds(&csr, 4, HDBSCAN_CAPI_METRIC_PRECOMPUTED,
            &configuration, &thresholds, &result) != HDBSCAN_CAPI_STATUS_OK) {
        return 32;
    }
    if (hdbscan_capi_result_view(result, &view) != HDBSCAN_CAPI_STATUS_OK
            || view.n_samples != 4 || view.n_clusters != 1
            || view.labels[0] != 0 || view.labels[1] != 0
            || view.labels[2] != 0 || view.labels[3] != 0) {
        hdbscan_capi_result_free(result);
        return 33;
    }
    hdbscan_capi_result_free(result);
    result = NULL;

    const double missing_features[] = {NAN, 0.0, 1.0, INFINITY, 5.0, 6.0};
    configuration.cluster_selection_method = HDBSCAN_CAPI_CLUSTER_SELECTION_EOM;
    if (hdbscan_capi_fit_dense_with_missing(missing_features, 6, 6, 1,
            HDBSCAN_CAPI_METRIC_EUCLIDEAN, &configuration, NULL, &result) != HDBSCAN_CAPI_STATUS_OK) {
        return 34;
    }
    if (hdbscan_capi_result_view(result, &view) != HDBSCAN_CAPI_STATUS_OK
            || view.n_samples != 6 || view.n_clusters != 2
            || view.labels[0] != -1 || view.labels[1] != 0 || view.labels[2] != 0
            || view.labels[3] != -1 || view.labels[4] != 1 || view.labels[5] != 1
            || view.probabilities[0] != 0.0 || view.probabilities[3] != 0.0) {
        hdbscan_capi_result_free(result);
        return 35;
    }
    double scores[6] = {-1.0};
    if (hdbscan_capi_result_outlier_scores(result, scores, 5) != HDBSCAN_CAPI_STATUS_INVALID_LENGTH
            || scores[0] != -1.0
            || hdbscan_capi_result_outlier_scores(result, scores, 6) != HDBSCAN_CAPI_STATUS_OK) {
        hdbscan_capi_result_free(result);
        return 40;
    }
    const double expected_scores[] = {1.0, 0.0, 0.0, 1.0, 0.0, 0.0};
    for (size_t i = 0; i < 6; ++i) {
        if (scores[i] != expected_scores[i]) {
            hdbscan_capi_result_free(result);
            return 41;
        }
    }
    int64_t cut_labels[6];
    if (hdbscan_capi_result_dbscan_clustering(result, 1.0, 2, cut_labels, 6)
            != HDBSCAN_CAPI_STATUS_OK) {
        hdbscan_capi_result_free(result);
        return 42;
    }
    for (size_t i = 0; i < 6; ++i) {
        if (cut_labels[i] != -1) {
            hdbscan_capi_result_free(result);
            return 43;
        }
    }
    if (hdbscan_capi_result_dbscan_clustering(result, 1.0 + DBL_EPSILON, 2, cut_labels, 6)
            != HDBSCAN_CAPI_STATUS_OK) {
        hdbscan_capi_result_free(result);
        return 44;
    }
    const int64_t expected_cut_labels[] = {-1, 0, 0, -1, 1, 1};
    for (size_t i = 0; i < 6; ++i) {
        if (cut_labels[i] != expected_cut_labels[i]) {
            hdbscan_capi_result_free(result);
            return 45;
        }
    }
    hdbscan_capi_result_free(result);
    result = NULL;

    const size_t missing_offsets[] = {0, 1, 2, 3, 4, 5, 6};
    const size_t missing_columns[] = {0, 0, 0, 0, 0, 0};
    const hdbscan_capi_csr_t missing_csr = {missing_offsets, missing_columns, missing_features, 6, 6};
    if (hdbscan_capi_fit_csr_with_missing(&missing_csr, 1, HDBSCAN_CAPI_METRIC_MANHATTAN,
            &configuration, NULL, &result) != HDBSCAN_CAPI_STATUS_OK) {
        return 36;
    }
    if (hdbscan_capi_result_view(result, &view) != HDBSCAN_CAPI_STATUS_OK
            || view.n_samples != 6 || view.n_clusters != 2
            || view.labels[0] != -1 || view.labels[3] != -1 || view.probabilities[5] != 1.0) {
        hdbscan_capi_result_free(result);
        return 37;
    }
    hdbscan_capi_result_free(result);
    result = NULL;

    if (hdbscan_capi_fit_dense_with_missing(missing_features, 2, 2, 1,
            HDBSCAN_CAPI_METRIC_EUCLIDEAN, &configuration, NULL, &result)
            != HDBSCAN_CAPI_STATUS_INVALID_SHAPE || result != NULL) {
        hdbscan_capi_result_free(result);
        return 38;
    }
    if (hdbscan_capi_fit_dense_configured(missing_features, 6, 6, 1,
            HDBSCAN_CAPI_METRIC_EUCLIDEAN, &configuration, &result)
            != HDBSCAN_CAPI_STATUS_NONFINITE_INPUT || result != NULL) {
        hdbscan_capi_result_free(result);
        return 39;
    }
    if (hdbscan_capi_fit_dense_prims(missing_features, 6, 6, 1,
            HDBSCAN_CAPI_METRIC_MANHATTAN, &configuration, NULL, &result) != HDBSCAN_CAPI_STATUS_OK) {
        return 46;
    }
    if (hdbscan_capi_result_view(result, &view) != HDBSCAN_CAPI_STATUS_OK
            || view.n_samples != 6 || view.n_clusters != 2
            || view.labels[0] != -1 || view.labels[1] != 0 || view.labels[5] != 1) {
        hdbscan_capi_result_free(result);
        return 47;
    }
    hdbscan_capi_result_free(result);
    result = NULL;
    if (hdbscan_capi_fit_dense_prims(missing_features, 6, 6, 1,
            HDBSCAN_CAPI_METRIC_COSINE, &configuration, NULL, &result)
            != HDBSCAN_CAPI_STATUS_INVALID_METRIC || result != NULL) {
        hdbscan_capi_result_free(result);
        return 48;
    }
    const hdbscan_capi_boruvka_options_t boruvka = {HDBSCAN_CAPI_BORUVKA_BALLTREE, 0, 3, 1};
    if (hdbscan_capi_fit_dense_boruvka(missing_features, 6, 6, 1,
            HDBSCAN_CAPI_METRIC_EUCLIDEAN, &configuration, NULL, &boruvka, &result) != HDBSCAN_CAPI_STATUS_OK) {
        return 49;
    }
    if (hdbscan_capi_result_view(result, &view) != HDBSCAN_CAPI_STATUS_OK
            || view.n_samples != 6 || view.n_clusters != 2
            || view.labels[0] != -1 || view.labels[1] != 0 || view.labels[5] != 1) {
        hdbscan_capi_result_free(result);
        return 50;
    }
    hdbscan_capi_result_free(result);
    const double train[] = {0.0, 0.1, 0.3, 1.0, 1.2, 10.0, 10.2, 10.6, 12.0, 18.0};
    const hdbscan_capi_options_t predict_options = {2, 2, 1.0};
    if (hdbscan_capi_fit_dense_euclidean(train, 10, 10, 1, &predict_options, &result)
            != HDBSCAN_CAPI_STATUS_OK) {
        return 51;
    }
    hdbscan_capi_prediction_model_t *prediction = NULL;
    if (hdbscan_capi_prediction_create(result, train, 9, 1, HDBSCAN_CAPI_METRIC_EUCLIDEAN, 2, &prediction)
            != HDBSCAN_CAPI_STATUS_INVALID_LENGTH || prediction != NULL
            || hdbscan_capi_prediction_create(result, train, 10, 1, HDBSCAN_CAPI_METRIC_COSINE, 2, &prediction)
            != HDBSCAN_CAPI_STATUS_INVALID_METRIC || prediction != NULL
            || hdbscan_capi_prediction_create(result, train, 10, 1, HDBSCAN_CAPI_METRIC_EUCLIDEAN, 2, &prediction)
            != HDBSCAN_CAPI_STATUS_OK || prediction == NULL) {
        hdbscan_capi_result_free(result);
        return 52;
    }
    hdbscan_capi_result_free(result);
    const double query[] = {0.0, 4.0, 10.35};
    hdbscan_capi_prediction_row_t rows[3] = {{-2, -1.0, -2}};
    double prediction_scores[3] = {-1.0, -1.0, -1.0};
    if (hdbscan_capi_approximate_predict(prediction, query, 2, rows, 3)
            != HDBSCAN_CAPI_STATUS_INVALID_LENGTH || rows[0].label != -2
            || hdbscan_capi_approximate_predict_scores(prediction, query, 2, prediction_scores, 3)
            != HDBSCAN_CAPI_STATUS_INVALID_LENGTH || prediction_scores[0] != -1.0
            || hdbscan_capi_approximate_predict(prediction, NULL, 0, NULL, 0)
            != HDBSCAN_CAPI_STATUS_OK) {
        hdbscan_capi_prediction_free(prediction);
        return 53;
    }
    if (hdbscan_capi_approximate_predict(prediction, query, 3, rows, 3) != HDBSCAN_CAPI_STATUS_OK
            || hdbscan_capi_approximate_predict_scores(prediction, query, 3, prediction_scores, 3) != HDBSCAN_CAPI_STATUS_OK
            || rows[0].label != 0 || rows[0].probability != 1.0 || rows[0].connecting_point != 0
            || rows[1].label != 0 || rows[1].probability != 0.08108108108108107 || rows[1].connecting_point != 4
            || rows[2].label != 1 || rows[2].probability != 1.0 || rows[2].connecting_point != 6
            || prediction_scores[0] != 0.0 || prediction_scores[1] != 0.918918918918919
            || prediction_scores[2] != -0.7142857142857145) {
        hdbscan_capi_prediction_free(prediction);
        return 54;
    }
    hdbscan_capi_membership_shape_t membership_shape = {0, 0};
    double query_memberships[6] = {-1.0};
    double training_memberships[20] = {-1.0};
    if (hdbscan_capi_membership_shape(prediction, &membership_shape) != HDBSCAN_CAPI_STATUS_OK
            || membership_shape.n_samples != 10 || membership_shape.n_clusters != 2
            || hdbscan_capi_membership_vector(prediction, query, 3, query_memberships, 5)
            != HDBSCAN_CAPI_STATUS_INVALID_LENGTH || query_memberships[0] != -1.0
            || hdbscan_capi_all_points_membership_vectors(prediction, training_memberships, 19)
            != HDBSCAN_CAPI_STATUS_INVALID_LENGTH || training_memberships[0] != -1.0) {
        hdbscan_capi_prediction_free(prediction);
        return 55;
    }
    if (hdbscan_capi_membership_vector(prediction, query, 3, query_memberships, 6)
            != HDBSCAN_CAPI_STATUS_OK
            || hdbscan_capi_all_points_membership_vectors(prediction, training_memberships, 20)
            != HDBSCAN_CAPI_STATUS_OK
            || query_memberships[0] != 0.999999997 || query_memberships[1] != 0.0
            || query_memberships[2] != 0.0475163036775151 || query_memberships[3] != 0.03356477716032272
            || query_memberships[4] != 0.0 || query_memberships[5] != 0.9999999940000001
            || training_memberships[0] != 1.0 || training_memberships[1] != 7.70098761457895e-310
            || training_memberships[18] != 0.023904315453994886
            || training_memberships[19] != 0.0571767656270861) {
        hdbscan_capi_prediction_free(prediction);
        return 56;
    }
    hdbscan_capi_prediction_free(prediction);
    hdbscan_capi_prediction_free(NULL);

    const double validity_data[] = {0.0, 1.0, 2.0, 5.0, 6.0, 8.0, 12.0, 15.0, 17.0, 27.0};
    const int64_t validity_labels[] = {0, 0, 0, 1, 1, 1, 3, 3, 3, -1};
    double validity_score = -1.0;
    double cluster_scores[4] = {-2.0};
    if (hdbscan_capi_validity_index(validity_data, 10, validity_labels, 10, 1,
            HDBSCAN_CAPI_METRIC_EUCLIDEAN, 0.0, 0, &validity_score, cluster_scores, 3)
            != HDBSCAN_CAPI_STATUS_INVALID_LENGTH
            || validity_score != -1.0 || cluster_scores[0] != -2.0) {
        return 57;
    }
    if (hdbscan_capi_validity_index(validity_data, 10, validity_labels, 10, 1,
            HDBSCAN_CAPI_METRIC_EUCLIDEAN, 0.0, 0, &validity_score, cluster_scores, 4)
            != HDBSCAN_CAPI_STATUS_OK || validity_score != 0.551
            || cluster_scores[0] != 0.7333333333333334 || cluster_scores[1] != 0.5199999999999999
            || !isnan(cluster_scores[2]) || cluster_scores[3] != 0.5833333333333334) {
        return 58;
    }
    double validity_distances[100];
    for (size_t i = 0; i < 10; ++i) {
        for (size_t j = 0; j < 10; ++j) {
            validity_distances[i * 10 + j] = fabs(validity_data[i] - validity_data[j]);
        }
    }
    if (hdbscan_capi_validity_index(validity_distances, 100, validity_labels, 10, 10,
            HDBSCAN_CAPI_METRIC_PRECOMPUTED, 1.0, 0, &validity_score, NULL, 0)
            != HDBSCAN_CAPI_STATUS_OK || validity_score != 0.551
            || hdbscan_capi_validity_index(validity_distances, 100, validity_labels, 10, 10,
                HDBSCAN_CAPI_METRIC_PRECOMPUTED, 1.0, 1, &validity_score, NULL, 0)
            != HDBSCAN_CAPI_STATUS_OK || validity_score != 0.62) {
        return 59;
    }
    const double robust_data[] = {0.0, 1.0, 3.0, 7.0, 9.0, 12.0, 21.0};
    const hdbscan_capi_robust_options_t robust_options = {7.1, 2, 1.0, 1, HDBSCAN_CAPI_ROBUST_GENERIC, 40, 1};
    int64_t robust_labels[7];
    hdbscan_capi_linkage_row_t robust_tree[6];
    if (hdbscan_capi_robust_single_linkage(robust_data, 7, 7, 1, HDBSCAN_CAPI_METRIC_EUCLIDEAN,
            &robust_options, robust_labels, 7, robust_tree, 6) != HDBSCAN_CAPI_STATUS_OK
            || robust_labels[0] != 0 || robust_labels[5] != 0 || robust_labels[6] != 1
            || robust_tree[0].left != 0 || robust_tree[0].right != 1 || robust_tree[0].distance != 3.0
            || robust_tree[5].left != 11 || robust_tree[5].right != 6
            || robust_tree[5].distance != 12.0 || robust_tree[5].size != 7) {
        return 60;
    }
    if (!check_branch_detection()) {
        return 61;
    }
    const double spatial_data[] = {0.0, 0.0, 1.0, 1.0, 6.0, 6.0, 7.0, 7.0};
    const hdbscan_capi_tree_distance_t minkowski = {HDBSCAN_CAPI_TREE_MINKOWSKI, 3.0, NULL, 0};
    const hdbscan_capi_fit_configuration_t spatial_fit = {
        {2, 1, 1.0}, HDBSCAN_CAPI_CLUSTER_SELECTION_EOM, 0, 0.0
    };
    const hdbscan_capi_boruvka_options_t spatial_search = {HDBSCAN_CAPI_BORUVKA_KDTREE, 0, 5, 1};
    if (hdbscan_capi_fit_dense_tree(spatial_data, 8, 4, 2, &minkowski,
            &spatial_fit, NULL, &spatial_search, &result) != HDBSCAN_CAPI_STATUS_OK) {
        return 62;
    }
    hdbscan_capi_result_free(result);
    const hdbscan_capi_robust_options_t spatial_robust = {1.5, 1, 1.0, 2, HDBSCAN_CAPI_ROBUST_PRIM, 5, 1};
    if (hdbscan_capi_robust_single_linkage_tree(spatial_data, 8, 4, 2, &minkowski, &spatial_robust,
            robust_labels, 4, robust_tree, 3) != HDBSCAN_CAPI_STATUS_OK
            || robust_labels[0] != 0 || robust_labels[3] != -1
            || robust_tree[0].distance != 1.2599210498948732) {
        return 63;
    }
    return 0;
}
