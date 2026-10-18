export interface DatasetUploadResponse {
  id: string;
  filename: string;
  file_size_bytes: number;
  row_count: number;
  col_count: number;
  columns: string[];
  suggested_target_column?: string;
  suggested_id_column?: string;
}

export interface ColumnProfile {
  name: string;
  data_type: string;
  missing_count: number;
  missing_percentage: number;
  unique_count: number;
  mean?: number | null;
  std?: number | null;
  min?: number | null;
  max?: number | null;
  categories_count?: number | null;
  top_category?: string | null;
  is_constant: boolean;
  is_potential_id: boolean;
  sample_values: any[];
}

export interface DatasetQuality {
  total_missing_values: number;
  duplicate_rows_count: number;
  constant_columns: string[];
  high_cardinality_categoricals: string[];
  potential_id_columns: string[];
  numerical_columns: string[];
  categorical_columns: string[];
  boolean_columns: string[];
  datetime_columns: string[];
  warnings: string[];
}

export interface DatasetOverview {
  row_count: number;
  col_count: number;
  missing_values_count: number;
  duplicate_rows_count: number;
  numerical_features_count: number;
  categorical_features_count: number;
  target_column?: string;
  id_column?: string;
}

export interface DatasetProfileResponse {
  overview: DatasetOverview;
  columns: ColumnProfile[];
  quality: DatasetQuality;
  data_preview: Record<string, any>[];
  preview_columns: string[];
}

export interface ClassDistributionItem {
  category: string;
  count: number;
  percentage: number;
}

export interface TargetStats {
  mean: number;
  std: number;
  min: number;
  max: number;
  median: number;
}

export interface ProblemDetectionResponse {
  target_column: string;
  detected_problem_type: 'classification' | 'regression';
  reason: string;
  unique_values_count: number;
  is_numeric: boolean;
  classes_count?: number;
  class_distribution?: ClassDistributionItem[];
  target_stats?: TargetStats;
  recommended_metrics: string[];
}

export interface PreprocessingSummaryResponse {
  experiment_id: string;
  numerical_features: string[];
  categorical_features: string[];
  target_column: string;
  id_column?: string;
  problem_type: string;
  train_samples_count: number;
  test_samples_count: number;
  features_count_after_encoding: number;
  configuration: Record<string, any>;
}

export interface ModelComparisonItem {
  id: string;
  model_name: string;
  model_type: string;
  metrics: Record<string, any>;
  training_time_sec: number;
  cv_mean?: number;
  cv_std?: number;
  rank: number;
}

export interface RecommendedModelInfo {
  model_id: string;
  model_name: string;
  primary_metric_name: string;
  primary_metric_value: number;
  rationale: string;
}

export interface ModelComparisonResponse {
  experiment_id: string;
  problem_type: string;
  models: ModelComparisonItem[];
  recommended_model?: RecommendedModelInfo;
  metric_keys: string[];
}

export interface DetailedEvaluationResponse {
  id: string;
  model_name: string;
  model_type: string;
  metrics: Record<string, any>;
  training_time_sec: number;
  hyperparameters?: Record<string, any>;
  cv_results?: Record<string, any>;
  confusion_matrix?: {
    labels: string[];
    matrix: number[][];
    normalized: number[][];
  };
  classification_report?: Record<string, any>;
  roc_curve?: {
    fpr: number[];
    tpr: number[];
    auc: number;
  };
  pr_curve?: {
    precision: number[];
    recall: number[];
    avg_precision: number;
  };
  actual_vs_predicted?: { actual: number; predicted: number }[];
  residuals?: {
    mean_residual: number;
    std_residual: number;
    distribution: { bin_start: number; bin_end: number; bin_center: number; count: number }[];
  };
}

export interface FeatureImportanceItem {
  feature: string;
  importance: number;
}

export interface ShapScatterPoint {
  feature: string;
  feature_value: number;
  shap_value: number;
}

export interface ShapSummaryResponse {
  model_id: string;
  model_name: string;
  explainer_type: string;
  feature_importance: FeatureImportanceItem[];
  shap_bar_data: FeatureImportanceItem[];
  scatter_samples: ShapScatterPoint[];
  base_value: number;
  feature_names: string[];
}

export interface FeatureContribution {
  feature: string;
  value: any;
  contribution: number;
}

export interface FeatureInputField {
  name: string;
  data_type: 'numeric' | 'categorical' | 'boolean';
  label: string;
  required: boolean;
  default_value?: any;
  min_value?: number;
  max_value?: number;
  step?: number;
  options?: string[];
}

export interface PredictionSchemaResponse {
  model_id: string;
  model_name: string;
  problem_type: string;
  target_column: string;
  fields: FeatureInputField[];
}

export interface PredictionResponse {
  model_id: string;
  prediction: any;
  probability?: number;
  feature_contributions: FeatureContribution[];
  base_value?: number;
}
