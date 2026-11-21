#include <inttypes.h>
#include <stdio.h>

#include "hdbscan_capi.h"

int main(void) {
    const double points[] = {0.0, 1.0, 5.0, 6.0};
    const size_t n_samples = sizeof(points) / sizeof(points[0]);
    const hdbscan_capi_options_t options = {2, 1, 1.0};
    hdbscan_capi_result_t *model = NULL;
    hdbscan_capi_status_t status = hdbscan_capi_fit_dense_euclidean(
        points, n_samples, n_samples, 1, &options, &model
    );
    if (status != HDBSCAN_CAPI_STATUS_OK) {
        fprintf(stderr, "HDBSCAN fit failed (%" PRIu32 ")\n", status);
        return 1;
    }
    hdbscan_capi_result_view_t view;
    status = hdbscan_capi_result_view(model, &view);
    if (status != HDBSCAN_CAPI_STATUS_OK) {
        hdbscan_capi_result_free(model);
        return 1;
    }
    printf("%zu clusters\n", view.n_clusters);
    for (size_t row = 0; row < view.n_samples; ++row) {
        printf("%zu %" PRId64 " %.3f\n", row, view.labels[row], view.probabilities[row]);
    }
    hdbscan_capi_result_free(model);
    return 0;
}
