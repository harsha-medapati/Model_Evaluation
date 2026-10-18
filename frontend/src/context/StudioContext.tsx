import React, { createContext, useContext, useState, useEffect } from 'react';
import {
  DatasetUploadResponse,
  DatasetProfileResponse,
  ProblemDetectionResponse,
  PreprocessingSummaryResponse,
  ModelComparisonResponse,
  DetailedEvaluationResponse,
  ShapSummaryResponse,
  PredictionSchemaResponse,
} from '../types';
import { apiClient } from '../api/client';

export type WorkflowStage =
  | 'dataset'
  | 'profile'
  | 'preprocess'
  | 'problem'
  | 'train'
  | 'compare'
  | 'evaluate'
  | 'explain'
  | 'predict';

interface StudioContextType {
  activeStage: WorkflowStage;
  setActiveStage: (stage: WorkflowStage) => void;
  completedStages: Set<WorkflowStage>;

  // Data
  dataset: DatasetUploadResponse | null;
  setDataset: (d: DatasetUploadResponse | null) => void;
  profile: DatasetProfileResponse | null;
  setProfile: (p: DatasetProfileResponse | null) => void;
  targetColumn: string;
  setTargetColumn: (col: string) => void;
  idColumn: string;
  setIdColumn: (col: string) => void;

  problemDetection: ProblemDetectionResponse | null;
  setProblemDetection: (p: ProblemDetectionResponse | null) => void;
  experimentSummary: PreprocessingSummaryResponse | null;
  setExperimentSummary: (e: PreprocessingSummaryResponse | null) => void;

  // Models & Training
  trainingRuns: any[];
  isTraining: boolean;
  setIsTraining: (val: boolean) => void;
  comparison: ModelComparisonResponse | null;
  setComparison: (c: ModelComparisonResponse | null) => void;
  selectedModelId: string | null;
  setSelectedModelId: (id: string | null) => void;
  selectedModelName: string;

  // Detailed & SHAP
  detailedEvaluation: DetailedEvaluationResponse | null;
  shapData: ShapSummaryResponse | null;
  predictionSchema: PredictionSchemaResponse | null;

  // Actions
  loadDatasetProfile: (target?: string, id?: string) => Promise<void>;
  runProblemDetection: (target: string) => Promise<void>;
  pollTrainingStatus: (experimentId: string) => Promise<void>;
  loadDetailedEvaluation: (modelId: string) => Promise<void>;
  loadShapData: (modelId: string) => Promise<void>;
  loadPredictionSchema: (modelId: string) => Promise<void>;
  markStageCompleted: (stage: WorkflowStage) => void;
}

const StudioContext = createContext<StudioContextType | undefined>(undefined);

export const StudioProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [activeStage, setActiveStage] = useState<WorkflowStage>('dataset');
  const [completedStages, setCompletedStages] = useState<Set<WorkflowStage>>(new Set(['dataset']));

  const [dataset, setDataset] = useState<DatasetUploadResponse | null>(null);
  const [profile, setProfile] = useState<DatasetProfileResponse | null>(null);
  const [targetColumn, setTargetColumn] = useState<string>('');
  const [idColumn, setIdColumn] = useState<string>('None');

  const [problemDetection, setProblemDetection] = useState<ProblemDetectionResponse | null>(null);
  const [experimentSummary, setExperimentSummary] = useState<PreprocessingSummaryResponse | null>(null);

  const [trainingRuns, setTrainingRuns] = useState<any[]>([]);
  const [isTraining, setIsTraining] = useState<boolean>(false);
  const [comparison, setComparison] = useState<ModelComparisonResponse | null>(null);
  const [selectedModelId, setSelectedModelId] = useState<string | null>(null);

  const [detailedEvaluation, setDetailedEvaluation] = useState<DetailedEvaluationResponse | null>(null);
  const [shapData, setShapData] = useState<ShapSummaryResponse | null>(null);
  const [predictionSchema, setPredictionSchema] = useState<PredictionSchemaResponse | null>(null);

  const markStageCompleted = (stage: WorkflowStage) => {
    setCompletedStages((prev) => new Set([...prev, stage]));
  };

  const loadDatasetProfile = async (target?: string, id?: string) => {
    if (!dataset) return;
    const tgt = target || targetColumn || dataset.suggested_target_column;
    const idC = id !== undefined ? id : idColumn || dataset.suggested_id_column || 'None';
    const prof = await apiClient.getDatasetProfile(dataset.id, tgt, idC === 'None' ? undefined : idC);
    setProfile(prof);
    if (prof.overview.target_column) setTargetColumn(prof.overview.target_column);
    if (prof.overview.id_column) setIdColumn(prof.overview.id_column);
  };

  const runProblemDetection = async (target: string) => {
    if (!dataset) return;
    const det = await apiClient.detectProblem(dataset.id, target);
    setProblemDetection(det);
  };

  const pollTrainingStatus = async (expId: string) => {
    setIsTraining(true);
    let attempts = 0;
    const maxAttempts = 60; // 60 * 1.5s = 90s max

    const interval = setInterval(async () => {
      attempts++;
      try {
        const runs = await apiClient.getTrainingStatus(expId);
        setTrainingRuns(runs);

        const allDone = runs.length > 0 && runs.every((r: any) => r.status === 'completed' || r.status === 'failed');
        if (allDone || attempts >= maxAttempts) {
          clearInterval(interval);
          setIsTraining(false);
          // Load comparison
          const comp = await apiClient.compareModels(expId);
          setComparison(comp);
          if (comp.models.length > 0) {
            const champion = comp.recommended_model?.model_id || comp.models[0].id;
            setSelectedModelId(champion);
          }
          markStageCompleted('train');
          markStageCompleted('compare');
          markStageCompleted('evaluate');
          markStageCompleted('explain');
          markStageCompleted('predict');
        }
      } catch (err) {
        console.error('Error polling training status:', err);
      }
    }, 1500);
  };

  const loadDetailedEvaluation = async (modelId: string) => {
    try {
      const evalData = await apiClient.getDetailedEvaluation(modelId);
      setDetailedEvaluation(evalData);
    } catch (err) {
      console.error('Failed to load detailed evaluation:', err);
    }
  };

  const loadShapData = async (modelId: string) => {
    try {
      const shap = await apiClient.getModelShap(modelId);
      setShapData(shap);
    } catch (err) {
      console.error('Failed to load SHAP data:', err);
    }
  };

  const loadPredictionSchema = async (modelId: string) => {
    try {
      const schema = await apiClient.getPredictionSchema(modelId);
      setPredictionSchema(schema);
    } catch (err) {
      console.error('Failed to load prediction schema:', err);
    }
  };

  // Sync selected model details when changed
  useEffect(() => {
    if (selectedModelId) {
      loadDetailedEvaluation(selectedModelId);
      loadShapData(selectedModelId);
      loadPredictionSchema(selectedModelId);
    }
  }, [selectedModelId]);

  const selectedModelName =
    comparison?.models.find((m) => m.id === selectedModelId)?.model_name ||
    detailedEvaluation?.model_name ||
    'Selected Model';

  return (
    <StudioContext.Provider
      value={{
        activeStage,
        setActiveStage,
        completedStages,
        dataset,
        setDataset,
        profile,
        setProfile,
        targetColumn,
        setTargetColumn,
        idColumn,
        setIdColumn,
        problemDetection,
        setProblemDetection,
        experimentSummary,
        setExperimentSummary,
        trainingRuns,
        isTraining,
        setIsTraining,
        comparison,
        setComparison,
        selectedModelId,
        setSelectedModelId,
        selectedModelName,
        detailedEvaluation,
        shapData,
        predictionSchema,
        loadDatasetProfile,
        runProblemDetection,
        pollTrainingStatus,
        loadDetailedEvaluation,
        loadShapData,
        loadPredictionSchema,
        markStageCompleted,
      }}
    >
      {children}
    </StudioContext.Provider>
  );
};

export const useStudio = () => {
  const context = useContext(StudioContext);
  if (!context) throw new Error('useStudio must be used within a StudioProvider');
  return context;
};
