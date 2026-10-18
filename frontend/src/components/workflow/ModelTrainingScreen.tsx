import React, { useState, useEffect } from 'react';
import { useStudio } from '../../context/StudioContext';
import { apiClient } from '../../api/client';
import {
  Cpu,
  Layers,
  CheckCircle2,
  AlertCircle,
  Loader2,
  Clock,
  ArrowRight,
  Sparkles,
} from 'lucide-react';

export const ModelTrainingScreen: React.FC = () => {
  const {
    experimentSummary,
    problemDetection,
    trainingRuns,
    isTraining,
    pollTrainingStatus,
    setActiveStage,
    markStageCompleted,
  } = useStudio();

  const isClassification = problemDetection?.detected_problem_type !== 'regression';
  const [activeTab, setActiveTab] = useState<'classification' | 'regression'>(
    isClassification ? 'classification' : 'regression'
  );

  const [availableModels, setAvailableModels] = useState<{
    classification: string[];
    regression: string[];
  }>({
    classification: [
      'Logistic Regression',
      'Decision Tree',
      'Random Forest',
      'Support Vector Machine (SVM)',
      'K-Nearest Neighbors (KNN)',
      'Naive Bayes',
      'XGBoost Classifier',
    ],
    regression: [
      'Linear Regression',
      'Ridge Regression',
      'Lasso Regression',
      'Decision Tree Regressor',
      'Random Forest Regressor',
      'Support Vector Regressor',
      'KNN Regressor',
      'Gradient Boosting Regressor',
      'XGBoost Regressor',
    ],
  });

  const [selectedModels, setSelectedModels] = useState<string[]>(
    isClassification
      ? [
          'Logistic Regression',
          'Decision Tree',
          'Random Forest',
          'Support Vector Machine (SVM)',
          'K-Nearest Neighbors (KNN)',
          'Naive Bayes',
          'XGBoost Classifier',
        ]
      : [
          'Linear Regression',
          'Ridge Regression',
          'Decision Tree Regressor',
          'Random Forest Regressor',
          'Gradient Boosting Regressor',
          'XGBoost Regressor',
        ]
  );

  const [cvFolds, setCvFolds] = useState<number>(5);
  const [useTuning, setUseTuning] = useState<boolean>(false);
  const [nJobs, setNJobs] = useState<number>(-1);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  useEffect(() => {
    apiClient
      .getAvailableModels()
      .then(setAvailableModels)
      .catch(console.error);
  }, []);

  const currentModelsList =
    activeTab === 'classification'
      ? availableModels.classification
      : availableModels.regression;

  const toggleModel = (name: string) => {
    if (selectedModels.includes(name)) {
      setSelectedModels(selectedModels.filter((m) => m !== name));
    } else {
      setSelectedModels([...selectedModels, name]);
    }
  };

  const handleSelectAll = () => {
    setSelectedModels(currentModelsList);
  };

  const handleDeselectAll = () => {
    setSelectedModels([]);
  };

  const handleStartTraining = async () => {
    if (!experimentSummary || selectedModels.length === 0) return;
    setErrorMsg(null);
    try {
      await apiClient.trainModels({
        experiment_id: experimentSummary.experiment_id,
        selected_models: selectedModels,
        cv_folds: cvFolds,
        use_hyperparameter_tuning: useTuning,
        n_jobs: nJobs,
      });
      // Start polling
      pollTrainingStatus(experimentSummary.experiment_id);
    } catch (err: any) {
      setErrorMsg(err.response?.data?.detail || 'Failed to start model training.');
    }
  };

  const allCompleted =
    trainingRuns.length > 0 &&
    trainingRuns.every((r) => r.status === 'completed' || r.status === 'failed');

  return (
    <div className="max-w-6xl mx-auto py-8 px-6 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-bold text-slate-900 tracking-tight">Train Multiple Models</h2>
          <p className="text-xs text-slate-500 mt-0.5">
            Train and evaluate candidate models using consistent cross-validation protocol
          </p>
        </div>

        {allCompleted && (
          <button
            onClick={() => {
              markStageCompleted('compare');
              setActiveStage('compare');
            }}
            className="flex items-center gap-2 px-5 py-2.5 bg-blue-600 hover:bg-blue-700 text-white font-semibold text-xs rounded-xl shadow transition"
          >
            View Model Comparison
            <ArrowRight className="w-4 h-4" />
          </button>
        )}
      </div>

      {errorMsg && (
        <div className="p-4 bg-red-50 border border-red-200 text-red-700 text-xs rounded-xl">
          {errorMsg}
        </div>
      )}

      {/* Tabs matching screenshot */}
      <div className="flex border-b border-slate-200">
        <button
          onClick={() => setActiveTab('classification')}
          className={`px-6 py-2.5 text-xs font-bold border-b-2 transition-all ${
            activeTab === 'classification'
              ? 'border-blue-600 text-blue-600'
              : 'border-transparent text-slate-500 hover:text-slate-700'
          }`}
        >
          Classification Models
        </button>
        <button
          onClick={() => setActiveTab('regression')}
          className={`px-6 py-2.5 text-xs font-bold border-b-2 transition-all ${
            activeTab === 'regression'
              ? 'border-blue-600 text-blue-600'
              : 'border-transparent text-slate-500 hover:text-slate-700'
          }`}
        >
          Regression Models
        </button>
      </div>

      {/* Main Grid: Checklist & Training Config */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Model Checklist */}
        <div className="lg:col-span-2 bg-white rounded-2xl border border-slate-200/80 p-6 shadow-xs space-y-4">
          <div className="flex items-center justify-between pb-2 border-b border-slate-100">
            <h3 className="text-xs font-bold text-slate-800 uppercase tracking-wider">
              Select Models to Train ({selectedModels.length} selected)
            </h3>
            <div className="flex gap-2 text-[11px]">
              <button
                type="button"
                onClick={handleSelectAll}
                className="text-blue-600 hover:underline font-semibold"
              >
                Select All
              </button>
              <span className="text-slate-300">•</span>
              <button
                type="button"
                onClick={handleDeselectAll}
                className="text-slate-500 hover:underline font-semibold"
              >
                Clear
              </button>
            </div>
          </div>

          <div className="space-y-2">
            {currentModelsList.map((modelName) => {
              const isChecked = selectedModels.includes(modelName);
              return (
                <label
                  key={modelName}
                  className={`flex items-center gap-3 p-3 rounded-xl border transition-all cursor-pointer ${
                    isChecked
                      ? 'bg-blue-50/50 border-blue-300 text-slate-900 font-semibold'
                      : 'bg-slate-50/50 border-slate-200/80 text-slate-600 hover:bg-slate-50'
                  }`}
                >
                  <input
                    type="checkbox"
                    checked={isChecked}
                    onChange={() => toggleModel(modelName)}
                    className="w-4 h-4 rounded text-blue-600 focus:ring-blue-500 border-slate-300"
                  />
                  <span className="text-xs">{modelName}</span>
                </label>
              );
            })}
          </div>
        </div>

        {/* Training Configuration Card matching reference image */}
        <div className="bg-white rounded-2xl border border-slate-200/80 p-6 shadow-xs space-y-5 h-fit">
          <h3 className="text-xs font-bold text-slate-800 uppercase tracking-wider pb-2 border-b border-slate-100">
            Training Configuration
          </h3>

          <div>
            <label className="block text-xs font-semibold text-slate-700 mb-1">
              Cross Validation Folds
            </label>
            <input
              type="number"
              min="2"
              max="10"
              value={cvFolds}
              onChange={(e) => setCvFolds(parseInt(e.target.value) || 5)}
              className="w-full bg-slate-50 border border-slate-200 rounded-xl px-3 py-2 text-xs font-medium text-slate-800 focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
            <span className="text-[10px] text-slate-400 mt-1 block">5 folds recommended</span>
          </div>

          <div>
            <div className="flex items-center justify-between mb-1">
              <label className="text-xs font-semibold text-slate-700">
                Use Hyperparameter Tuning
              </label>
              <input
                type="checkbox"
                checked={useTuning}
                onChange={(e) => setUseTuning(e.target.checked)}
                className="w-4 h-4 rounded text-blue-600"
              />
            </div>
            <span className="text-[10px] text-slate-400 block">
              GridSearchCV optimization for hyperparameters
            </span>
          </div>

          <div>
            <label className="block text-xs font-semibold text-slate-700 mb-1">
              n_jobs (parallel processing)
            </label>
            <select
              value={nJobs}
              onChange={(e) => setNJobs(parseInt(e.target.value))}
              className="w-full bg-slate-50 border border-slate-200 rounded-xl px-3 py-2 text-xs font-medium text-slate-800 focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="-1">-1 (Use all available CPU cores)</option>
              <option value="2">2 threads</option>
              <option value="4">4 threads</option>
              <option value="1">1 (Single threaded)</option>
            </select>
          </div>

          <div className="pt-2">
            <button
              type="button"
              disabled={isTraining || selectedModels.length === 0}
              onClick={handleStartTraining}
              className="w-full flex items-center justify-center gap-2 py-3 bg-blue-600 hover:bg-blue-700 disabled:opacity-50 text-white font-bold text-xs rounded-xl shadow-md transition"
            >
              {isTraining ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  Training Models in Background...
                </>
              ) : (
                <>
                  Start Training
                  <ArrowRight className="w-4 h-4" />
                </>
              )}
            </button>
          </div>
        </div>
      </div>

      {/* Live Training Status List */}
      {trainingRuns.length > 0 && (
        <div className="bg-white rounded-2xl border border-slate-200/80 p-6 shadow-xs space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-xs font-bold text-slate-800 uppercase tracking-wider flex items-center gap-2">
              <Clock className="w-4 h-4 text-slate-500" />
              Training Execution Progress
            </h3>
            {isTraining && (
              <span className="flex items-center gap-1.5 text-xs text-blue-600 font-semibold">
                <Loader2 className="w-3.5 h-3.5 animate-spin" />
                Training running asynchronously...
              </span>
            )}
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
            {trainingRuns.map((r) => {
              const isDone = r.status === 'completed';
              const isRunning = r.status === 'training';
              const isFailed = r.status === 'failed';

              return (
                <div
                  key={r.id}
                  className="p-4 rounded-xl border border-slate-200/80 bg-slate-50/60 flex items-center justify-between"
                >
                  <div>
                    <div className="font-semibold text-xs text-slate-800">{r.model_name}</div>
                    <div className="text-[10px] text-slate-400 mt-0.5">
                      {isDone && r.training_time_sec ? `${r.training_time_sec}s` : r.status}
                    </div>
                  </div>

                  <div>
                    {isDone && (
                      <span className="flex items-center gap-1 px-2 py-0.5 bg-emerald-100 text-emerald-800 text-[10px] font-bold rounded-full">
                        <CheckCircle2 className="w-3 h-3 text-emerald-600" />
                        Completed
                      </span>
                    )}
                    {isRunning && (
                      <span className="flex items-center gap-1 px-2 py-0.5 bg-blue-100 text-blue-800 text-[10px] font-bold rounded-full">
                        <Loader2 className="w-3 h-3 animate-spin text-blue-600" />
                        Training
                      </span>
                    )}
                    {isFailed && (
                      <span className="flex items-center gap-1 px-2 py-0.5 bg-rose-100 text-rose-800 text-[10px] font-bold rounded-full">
                        <AlertCircle className="w-3 h-3 text-rose-600" />
                        Failed
                      </span>
                    )}
                    {r.status === 'queued' && (
                      <span className="px-2 py-0.5 bg-slate-200 text-slate-600 text-[10px] font-bold rounded-full">
                        Queued
                      </span>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
};
