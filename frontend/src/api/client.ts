import axios from 'axios';
import {
  DatasetUploadResponse,
  DatasetProfileResponse,
  ProblemDetectionResponse,
  PreprocessingSummaryResponse,
  ModelComparisonResponse,
  DetailedEvaluationResponse,
  ShapSummaryResponse,
  PredictionSchemaResponse,
  PredictionResponse
} from '../types';

const api = axios.create({
  baseURL: '/api/v1',
  headers: {
    'Content-Type': 'application/json',
  },
});

export const apiClient = {
  // Datasets
  async uploadDataset(file: File): Promise<DatasetUploadResponse> {
    const formData = new FormData();
    formData.append('file', file);
    const res = await api.post<DatasetUploadResponse>('/datasets/upload', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return res.data;
  },

  async listSampleDatasets(): Promise<{ filename: string; size_bytes: number; description: string }[]> {
    const res = await api.get('/datasets/samples');
    return res.data;
  },

  async loadSampleDataset(filename: string): Promise<DatasetUploadResponse> {
    const res = await api.post<DatasetUploadResponse>(`/datasets/load-sample?filename=${encodeURIComponent(filename)}`);
    return res.data;
  },

  async getDatasetProfile(datasetId: string, targetCol?: string, idCol?: string): Promise<DatasetProfileResponse> {
    const params = new URLSearchParams();
    if (targetCol) params.append('target_column', targetCol);
    if (idCol) params.append('id_column', idCol);
    const res = await api.get<DatasetProfileResponse>(`/datasets/${datasetId}/profile?${params.toString()}`);
    return res.data;
  },

  // Preprocessing & Problem Detection
  async detectProblem(datasetId: string, targetCol: string): Promise<ProblemDetectionResponse> {
    const res = await api.get<ProblemDetectionResponse>(
      `/preprocessing/detect?dataset_id=${encodeURIComponent(datasetId)}&target_column=${encodeURIComponent(targetCol)}`
    );
    return res.data;
  },

  async configurePreprocessing(payload: {
    dataset_id: string;
    target_column: string;
    id_column?: string;
    numerical_imputer: string;
    categorical_imputer: string;
    categorical_encoding: string;
    feature_scaling: string;
    test_size: number;
    random_state: number;
  }): Promise<PreprocessingSummaryResponse> {
    const res = await api.post<PreprocessingSummaryResponse>('/preprocessing/configure', payload);
    return res.data;
  },

  // Models & Training
  async getAvailableModels(): Promise<{ classification: string[]; regression: string[] }> {
    const res = await api.get('/models/available');
    return res.data;
  },

  async trainModels(payload: {
    experiment_id: string;
    selected_models: string[];
    cv_folds: number;
    use_hyperparameter_tuning: boolean;
    n_jobs: number;
  }): Promise<{ message: string; run_ids: string[] }> {
    const res = await api.post('/models/train', payload);
    return res.data;
  },

  async getTrainingStatus(experimentId: string): Promise<any[]> {
    const res = await api.get(`/models/status/${experimentId}`);
    return res.data;
  },

  async compareModels(experimentId: string, sortMetric?: string): Promise<ModelComparisonResponse> {
    const param = sortMetric ? `?sort_metric=${encodeURIComponent(sortMetric)}` : '';
    const res = await api.get<ModelComparisonResponse>(`/models/compare/${experimentId}${param}`);
    return res.data;
  },

  async getDetailedEvaluation(modelRunId: string): Promise<DetailedEvaluationResponse> {
    const res = await api.get<DetailedEvaluationResponse>(`/models/${modelRunId}/evaluation`);
    return res.data;
  },

  // Explainability (SHAP)
  async getModelShap(modelRunId: string): Promise<ShapSummaryResponse> {
    const res = await api.get<ShapSummaryResponse>(`/explain/${modelRunId}`);
    return res.data;
  },

  // Prediction
  async getPredictionSchema(modelRunId: string): Promise<PredictionSchemaResponse> {
    const res = await api.get<PredictionSchemaResponse>(`/predictions/${modelRunId}/schema`);
    return res.data;
  },

  async makePrediction(modelId: string, features: Record<string, any>): Promise<PredictionResponse> {
    const res = await api.post<PredictionResponse>('/predictions/', {
      model_id: modelId,
      features,
    });
    return res.data;
  },

  // Export URLs
  getModelExportUrl(modelRunId: string): string {
    return `/api/v1/exports/model/${modelRunId}`;
  },

  getComparisonCsvUrl(experimentId: string): string {
    return `/api/v1/exports/csv/${experimentId}`;
  },

  getPdfReportUrl(experimentId: string): string {
    return `/api/v1/exports/pdf/${experimentId}`;
  },
};
