use hdbscan_core::{
    BoruvkaOptions, BoruvkaTree, ClusterSelection, ClusterSelectionMethod,
    ClusterSelectionThresholds, CondensedEdge, FeatureMetric, FitOptions, MembershipError,
    MembershipQuery, PredictionModel, TreeDistance, dbscan_clustering,
    fit_csr_manhattan_f32_with_missing, fit_csr_precomputed_configured, fit_csr_with_missing,
    fit_dense_boruvka, fit_dense_precomputed_configured, fit_dense_prims, fit_dense_with_missing,
    outlier_scores,
    validity::{
        RelativeValidityError, RelativeValidityTree, ValidityError, relative_validity,
        validity_index_dense, validity_index_precomputed,
    },
};
use serde_json::Value;

fn bits(record: &Value) -> Vec<u64> {
    record["bits"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| u64::from_str_radix(value.as_str().unwrap(), 16).unwrap())
        .collect()
}

fn integers(record: &Value) -> Vec<i64> {
    record["values"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_i64().unwrap())
        .collect()
}

fn input_float(value: &Value) -> f64 {
    match value.as_str() {
        Some("inf") => f64::INFINITY,
        Some("-inf") => f64::NEG_INFINITY,
        Some("nan") => f64::NAN,
        _ => value.as_f64().unwrap(),
    }
}

fn unpack_matrix(record: &Value) -> Vec<f64> {
    match record["dtype"].as_str().unwrap() {
        "float32" => bits(record)
            .into_iter()
            .map(|value| f64::from(f32::from_bits(value as u32)))
            .collect(),
        "float64" => bits(record).into_iter().map(f64::from_bits).collect(),
        dtype => panic!("unexpected feature dtype {dtype}"),
    }
}

#[test]
fn validity_matches_the_public_reference() {
    let fixture = if cfg!(target_os = "windows") {
        include_str!("../../../reference/fixtures/validity-0.8.44-windows-x86_64.json")
    } else {
        include_str!("../../../reference/fixtures/validity-0.8.44-linux-x86_64.json")
    };
    let fixture: Value = serde_json::from_str(fixture).unwrap();
    for (name, case) in fixture["cases"].as_object().unwrap() {
        let input = &case["input"];
        let options = &case["options"];
        if input["dtype"] != "float64"
            || !options["n_jobs"].is_null()
            || name == "precomputed-raw-layout-error"
        {
            continue;
        }
        let dimension = input["shape"][1].as_u64().unwrap() as usize;
        let raw = options["mst_raw_dist"].as_bool().unwrap_or_default();
        let metric = options["metric"].as_str().unwrap_or("euclidean");
        let feature_metric = match metric {
            "euclidean" => FeatureMetric::Euclidean,
            "cityblock" => FeatureMetric::Manhattan,
            "cosine" => FeatureMetric::Cosine,
            "precomputed" => FeatureMetric::Euclidean,
            _ => continue,
        };
        let values = unpack_matrix(input);
        let labels = integers(&case["labels"]);
        let observed = if metric == "precomputed" {
            let Some(dimension) = options["d"].as_f64() else {
                continue;
            };
            validity_index_precomputed(&values, &labels, dimension, raw)
        } else {
            validity_index_dense(&values, dimension, &labels, feature_metric, raw)
        };
        if let Some(message) = case["error"]["message"].as_str() {
            let error = observed.unwrap_err();
            if case["error"]["type"] == "AssertionError" {
                assert_eq!(error, ValidityError::MissingTreeNeighbor, "{name}");
            } else {
                assert_eq!(error.to_string(), message, "{name}");
            }
            continue;
        }
        let scores = observed.unwrap_or_else(|error| panic!("{name}: {error}"));
        let expected = &case["result"][0];
        let overall = if expected["dtype"] == "float64" {
            bits(expected)[0]
        } else {
            0.0_f64.to_bits()
        };
        assert_eq!(scores.overall.to_bits(), overall, "{name}");
        let expected = case["result"][1]["bits"].as_array().unwrap();
        assert_eq!(scores.per_cluster.len(), expected.len(), "{name}");
        for (actual, record) in scores.per_cluster.iter().zip(expected) {
            if let Some(digits) = record.as_str() {
                assert_eq!(
                    actual.to_bits(),
                    u64::from_str_radix(digits, 16).unwrap(),
                    "{name}"
                );
            } else {
                assert!(actual.is_nan(), "{name}");
            }
        }
    }
}

#[test]
fn memberships_match_the_public_reference() {
    let fixture = if cfg!(target_os = "windows") {
        include_str!("../../../reference/fixtures/prediction-0.8.44-windows-x86_64.json")
    } else {
        include_str!("../../../reference/fixtures/prediction-0.8.44-linux-x86_64.json")
    };
    let fixture: Value = serde_json::from_str(fixture).unwrap();
    for (name, case) in fixture["cases"].as_object().unwrap() {
        let tree = &case["condensed"];
        let condensed: Vec<_> = integers(&tree["parent"])
            .into_iter()
            .zip(integers(&tree["child"]))
            .zip(bits(&tree["lambda_val"]))
            .zip(integers(&tree["child_size"]))
            .map(|(((parent, child), lambda), size)| CondensedEdge {
                parent: parent as usize,
                child: child as usize,
                lambda: f64::from_bits(lambda),
                child_size: size as usize,
            })
            .collect();
        let dimensions = case["input"]["shape"][1].as_u64().unwrap() as usize;
        let value = |key: &str| {
            case["parameter_overrides"]
                .get(key)
                .unwrap_or(&fixture["parameters"][key])
        };
        let min_samples = value("min_samples").as_u64().unwrap() as usize;
        let metric = match value("metric").as_str() {
            Some("manhattan" | "cityblock" | "l1") => TreeDistance::Manhattan,
            Some("chebyshev" | "infinity") => TreeDistance::Chebyshev,
            Some("minkowski" | "p") => TreeDistance::Minkowski {
                p: 2.0,
                weights: None,
            },
            _ => TreeDistance::Euclidean,
        };
        let model = PredictionModel::new_with_distance(
            unpack_matrix(&case["input"]),
            dimensions,
            metric,
            min_samples,
            &integers(&case["labels"]),
            &condensed,
        )
        .unwrap();
        assert_eq!(
            model
                .core_distances()
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            bits(&case["core_distances"]),
            "{name} core distances",
        );
        let model = if case["input"]["layout"] == "fortran" {
            model.with_column_major_training()
        } else {
            model
        };
        for (kind, output) in [
            (
                "new",
                model.membership_vector_for_layout(MembershipQuery {
                    values: &unpack_matrix(&case["queries"]),
                    min_samples,
                    column_major: case["queries"]["layout"] == "fortran",
                }),
            ),
            ("training", model.all_points_membership_vectors()),
        ] {
            let expected = &case["membership"][kind];
            match expected["error"]["type"].as_str() {
                Some("KeyError") => assert!(
                    matches!(output, Err(MembershipError::MissingDensity(node))
                        if Some(node as u64) == expected["error"]["args"][0].as_u64()),
                    "{name}/{kind}: {output:?}",
                ),
                Some("ValueError") => assert!(
                    matches!(output, Err(MembershipError::NoClusters)),
                    "{name}/{kind}: {output:?}"
                ),
                None => {
                    let matrix = output.unwrap_or_else(|error| panic!("{name}/{kind}: {error}"));
                    let shape = expected["array"]["shape"].as_array().unwrap();
                    assert_eq!(
                        matrix.rows as u64,
                        shape[0].as_u64().unwrap(),
                        "{name}/{kind}"
                    );
                    if shape.len() == 1 {
                        assert_eq!(matrix.clusters, 0, "{name}/{kind}");
                        assert!(matrix.values.is_empty(), "{name}/{kind}");
                        continue;
                    }
                    assert_eq!(
                        matrix.clusters as u64,
                        shape[1].as_u64().unwrap(),
                        "{name}/{kind}"
                    );
                    let expected = bits(&expected["array"]);
                    assert_eq!(matrix.values.len(), expected.len(), "{name}/{kind}");
                    for (index, (value, expected)) in
                        matrix.values.into_iter().zip(expected).enumerate()
                    {
                        assert_eq!(value.to_bits(), expected, "{name}/{kind} element {index}");
                    }
                }
                Some(other) => panic!("{name}/{kind}: unexpected reference exception {other}"),
            }
        }
    }
}

#[test]
fn fit_matches_the_public_reference() {
    let (platform, fixture) = if cfg!(target_env = "msvc") {
        (
            "Windows",
            include_str!("../../../reference/fixtures/hdbscan-0.8.44-windows-x86_64.json"),
        )
    } else {
        (
            "Linux",
            include_str!("../../../reference/fixtures/hdbscan-0.8.44-linux-x86_64.json"),
        )
    };
    let fixture: Value = serde_json::from_str(fixture).unwrap();
    for (case, record) in fixture["cases"].as_object().unwrap() {
        let parameter = |name: &str| {
            record["parameter_overrides"]
                .get(name)
                .unwrap_or(&fixture["parameters"][name])
        };
        let options = FitOptions {
            min_cluster_size: parameter("min_cluster_size").as_u64().unwrap() as usize,
            min_samples: parameter("min_samples").as_u64().map(|n| n as usize),
            alpha: parameter("alpha").as_f64().unwrap(),
        }
        .with_cluster_selection(ClusterSelection {
            method: match parameter("cluster_selection_method").as_str().unwrap() {
                "eom" => ClusterSelectionMethod::Eom,
                "leaf" => ClusterSelectionMethod::Leaf,
                other => panic!("unexpected reference cluster selection method {other}"),
            },
            allow_single_cluster: parameter("allow_single_cluster").as_bool().unwrap(),
        })
        .with_cluster_selection_thresholds(ClusterSelectionThresholds {
            epsilon: parameter("cluster_selection_epsilon").as_f64().unwrap(),
            epsilon_max: parameter("cluster_selection_epsilon_max")
                .as_f64()
                .unwrap_or(f64::INFINITY),
            persistence: parameter("cluster_selection_persistence").as_f64().unwrap(),
            max_cluster_size: parameter("max_cluster_size").as_f64().unwrap(),
        });
        let input = &record["input"];
        let n_samples = input["shape"][0].as_u64().unwrap() as usize;
        let n_features = input["shape"][1].as_u64().unwrap() as usize;
        let metric = match parameter("metric").as_str().unwrap() {
            "euclidean" => Some(FeatureMetric::Euclidean),
            "manhattan" => Some(FeatureMetric::Manhattan),
            "cosine" => Some(FeatureMetric::Cosine),
            "precomputed" => None,
            other => panic!("unexpected reference metric {other}"),
        };
        let result = if input["format"] == "csr" {
            let values: Vec<_> = input["values"]
                .as_array()
                .unwrap()
                .iter()
                .map(input_float)
                .collect();
            let indices = |name: &str| {
                input[name]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_u64().unwrap() as usize)
                    .collect::<Vec<_>>()
            };
            let indptr = indices("indptr");
            let columns = indices("indices");
            match metric {
                Some(FeatureMetric::Manhattan) if input["dtype"] == "float32" => {
                    let values: Vec<_> = values.iter().map(|&value| value as f32).collect();
                    fit_csr_manhattan_f32_with_missing(
                        &indptr, &columns, &values, n_features, options,
                    )
                }
                Some(metric) => {
                    fit_csr_with_missing(&indptr, &columns, &values, n_features, metric, options)
                }
                None => fit_csr_precomputed_configured(
                    &indptr,
                    &columns,
                    &values,
                    options,
                    parameter("max_dist").as_f64().unwrap_or(0.0),
                ),
            }
        } else {
            let values: Vec<_> = input["values"]
                .as_array()
                .unwrap()
                .iter()
                .flat_map(|row| row.as_array().unwrap())
                .map(input_float)
                .collect();
            let algorithm = parameter("algorithm").as_str().unwrap();
            let use_prims = matches!(algorithm, "prims_kdtree" | "prims_balltree")
                || algorithm == "best" && n_features > 60;
            let use_boruvka = matches!(algorithm, "boruvka_kdtree" | "boruvka_balltree")
                || algorithm == "best" && n_features <= 60;
            match metric {
                Some(metric) if use_prims => {
                    fit_dense_prims(&values, n_samples, n_features, metric, options)
                }
                Some(metric) if use_boruvka => fit_dense_boruvka(
                    &values,
                    n_samples,
                    n_features,
                    metric,
                    options,
                    BoruvkaOptions {
                        tree: if algorithm == "boruvka_balltree" {
                            BoruvkaTree::BallTree
                        } else {
                            BoruvkaTree::KdTree
                        },
                        approximate: parameter("approx_min_span_tree").as_bool().unwrap(),
                        leaf_size: parameter("leaf_size").as_u64().unwrap_or(40) as usize,
                        core_dist_n_jobs: parameter("core_dist_n_jobs").as_i64().unwrap(),
                    },
                ),
                Some(metric) => {
                    fit_dense_with_missing(&values, n_samples, n_features, metric, options)
                }
                None => {
                    assert_eq!(n_samples, n_features);
                    fit_dense_precomputed_configured(&values, n_samples, options)
                }
            }
        }
        .unwrap_or_else(|error| panic!("{platform} / {case}: {error}"));
        let context = format!("{platform} / {case}");
        let output = &record["outputs"];
        let scores: Vec<_> = outlier_scores(&result.condensed_tree)
            .unwrap()
            .into_iter()
            .map(f64::to_bits)
            .collect();
        assert_eq!(scores, bits(&output["outlier_scores_"]), "{context} GLOSH");
        let relative = &output["relative_validity_"];
        let tree = if metric.is_none() && input["format"] == "csr" {
            RelativeValidityTree::CsrPrecomputed
        } else {
            RelativeValidityTree::EdgeList
        };
        let observed = relative_validity(&result.labels, &result.minimum_spanning_tree, tree);
        if relative["error"]["type"] == "ValueError" {
            assert_eq!(
                observed,
                Err(RelativeValidityError::SparseTreeShape),
                "{context} relative validity"
            );
        } else {
            assert_eq!(
                observed.unwrap().to_bits(),
                bits(&relative["result"])[0],
                "{context} relative validity"
            );
        }
        for cut in output["dbscan_cuts"].as_array().unwrap() {
            let distance = f64::from_bits(
                u64::from_str_radix(cut["cut_distance_bits"].as_str().unwrap(), 16).unwrap(),
            );
            let minimum = cut["min_cluster_size"].as_i64().unwrap();
            let labels = dbscan_clustering(&result.single_linkage_tree, distance, minimum).unwrap();
            assert_eq!(
                labels,
                integers(&cut["labels"]),
                "{context} DBSCAN* at {distance:?} with min_cluster_size={minimum}"
            );
        }
        let mst: Vec<_> = result
            .minimum_spanning_tree
            .iter()
            .flat_map(|edge| [edge.source as f64, edge.target as f64, edge.distance])
            .map(f64::to_bits)
            .collect();
        assert_eq!(
            mst,
            bits(&output["minimum_spanning_tree_"]),
            "{context} MST"
        );
        let linkage: Vec<_> = result
            .single_linkage_tree
            .iter()
            .flat_map(|edge| {
                [
                    edge.left as f64,
                    edge.right as f64,
                    edge.distance,
                    edge.size as f64,
                ]
            })
            .map(f64::to_bits)
            .collect();
        assert_eq!(
            linkage,
            bits(&output["single_linkage_tree_"]),
            "{context} linkage"
        );
        let fields = &output["condensed_tree_"]["fields"];
        for (column, values) in [
            (
                0,
                result
                    .condensed_tree
                    .iter()
                    .map(|e| e.parent as i64)
                    .collect::<Vec<_>>(),
            ),
            (
                1,
                result
                    .condensed_tree
                    .iter()
                    .map(|e| e.child as i64)
                    .collect(),
            ),
            (
                3,
                result
                    .condensed_tree
                    .iter()
                    .map(|e| e.child_size as i64)
                    .collect(),
            ),
        ] {
            assert_eq!(
                values,
                integers(&fields[column]),
                "{context} condensed tree column {column}"
            );
        }
        let lambdas: Vec<_> = result
            .condensed_tree
            .iter()
            .map(|e| e.lambda.to_bits())
            .collect();
        assert_eq!(lambdas, bits(&fields[2]), "{context} condensed tree lambda");
        assert_eq!(
            result.labels,
            integers(&output["labels_"]),
            "{context} labels"
        );
        for (name, values) in [
            ("probabilities_", result.probabilities),
            ("cluster_persistence_", result.cluster_persistence),
        ] {
            let actual: Vec<_> = values.into_iter().map(f64::to_bits).collect();
            assert_eq!(actual, bits(&output[name]), "{context} {name}");
        }
    }
}
