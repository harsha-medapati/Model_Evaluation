import React, { useState } from 'react';
import { useStudio } from '../../context/StudioContext';
import { apiClient } from '../../api/client';
import {
  Sliders,
  CheckCircle,
  HelpCircle,
  ArrowRight,
  Loader2,
  Sparkles,
} from 'lucide-react';

export const PreprocessingScreen: React.FC = () => {
  const {
    dataset,
    profile,
    targetColumn,
    idColumn,
    problemDetection,
    setExperimentSummary,
    setActiveStage,
    markStageCompleted,
  } = useStudio();

  const [numImputer, setNumImputer] = useState('median');
  const [catImputer, setCatImputer] = useState('most_frequent');
  const [encoding, setEncoding] = useState('onehot');
  const [scaling, setScaling] = useState('standard');
  const [testSize, setTestSize] = useState(0.2);
  const [randomState, setRandomState] = useState(42);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [stratify, setStratify] = useState(true);

  const [isApplying, setIsApplying] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  if (!dataset || !profile) {
    return (
      <div className="p-8 text-center text-slate-500">
        Please upload and analyze a dataset first.
      </div>
    );
  }

  const numCols = profile.quality.numerical_columns.filter((c) => c !== targetColumn && c !== idColumn);
  const catCols = profile.quality.categorical_columns.filter((c) => c !== targetColumn && c !== idColumn);

  const handleApply = async () => {
    setIsApplying(true);
    setErrorMsg(null);
    try {
      const res = await apiClient.configurePreprocessing({
        dataset_id: dataset.id,
        target_column: targetColumn,
        id_column: idColumn === 'None' ? undefined : idColumn,
        numerical_imputer: numImputer,
        categorical_imputer: catImputer,
        categorical_encoding: encoding,
        feature_scaling: scaling,
        test_size: testSize,
        random_state: randomState,
      });
      setExperimentSummary(res);
      markStageCompleted('preprocess');
      markStageCompleted('problem');
      setActiveStage('problem');
    } catch (err: any) {
      setErrorMsg(err.response?.data?.detail || 'Failed to apply preprocessing pipeline.');
    } finally {
      setIsApplying(false);
    }
  };

  return (
    <div className="max-w-6xl mx-auto py-8 px-6 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-bold text-slate-900 tracking-tight">Preprocessing Configuration</h2>
          <p className="text-xs text-slate-500 mt-0.5">
            Configure imputation, encoding, feature scaling, and train/test validation protocol
          </p>
        </div>
      </div>

      {errorMsg && (
        <div className="p-4 bg-red-50 border border-red-200 text-red-700 text-xs rounded-xl">
          {errorMsg}
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Form: Preprocessing controls */}
        <div className="lg:col-span-2 bg-white rounded-2xl border border-slate-200/80 p-6 shadow-xs space-y-5">
          {/* Missing Value Strategy */}
          <div>
            <label className="block text-xs font-bold text-slate-800 mb-1.5 flex items-center justify-between">
              <span>Missing Value Strategy</span>
              <span className="text-[10px] text-slate-400 font-normal">Adapts to data types</span>
            </label>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              <div>
                <span className="text-[11px] font-medium text-slate-500 mb-1 block">Numerical Features</span>
                <select
                  value={numImputer}
                  onChange={(e) => setNumImputer(e.target.value)}
                  className="w-full bg-slate-50 border border-slate-200 rounded-xl px-3 py-2 text-xs font-medium text-slate-800 focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  <option value="median">Median (robust to outliers)</option>
                  <option value="mean">Mean</option>
                  <option value="most_frequent">Most Frequent</option>
                  <option value="constant">Constant (0.0)</option>
                </select>
              </div>

              <div>
                <span className="text-[11px] font-medium text-slate-500 mb-1 block">Categorical Features</span>
                <select
                  value={catImputer}
                  onChange={(e) => setCatImputer(e.target.value)}
                  className="w-full bg-slate-50 border border-slate-200 rounded-xl px-3 py-2 text-xs font-medium text-slate-800 focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  <option value="most_frequent">Most Frequent (mode)</option>
                  <option value="constant">Constant ("Missing")</option>
                </select>
              </div>
            </div>
          </div>

          {/* Categorical Encoding */}
          <div>
            <label className="block text-xs font-bold text-slate-800 mb-1.5">
              Categorical Encoding
            </label>
            <select
              value={encoding}
              onChange={(e) => setEncoding(e.target.value)}
              className="w-full bg-slate-50 border border-slate-200 rounded-xl px-3 py-2 text-xs font-medium text-slate-800 focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="onehot">One Hot Encoding (Standard for nominal categories)</option>
              <option value="ordinal">Ordinal Encoding (Ordered categories)</option>
            </select>
            <p className="text-[11px] text-slate-400 mt-1">
              Encodes categorical variables into numerical matrices, ignoring unseen categories at test/inference time.
            </p>
          </div>

          {/* Feature Scaling */}
          <div>
            <label className="block text-xs font-bold text-slate-800 mb-1.5">
              Feature Scaling
            </label>
            <select
              value={scaling}
              onChange={(e) => setScaling(e.target.value)}
              className="w-full bg-slate-50 border border-slate-200 rounded-xl px-3 py-2 text-xs font-medium text-slate-800 focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="standard">Standard Scaler (zero mean, unit variance)</option>
              <option value="minmax">MinMax Scaler (bounded 0 to 1)</option>
              <option value="robust">Robust Scaler (median and interquartile range)</option>
              <option value="none">No Scaling</option>
            </select>
            <p className="text-[11px] text-slate-400 mt-1">
              Fitted strictly on training data to prevent train/test data leakage.
            </p>
          </div>

          {/* Test Size Slider */}
          <div className="pt-2">
            <div className="flex items-center justify-between text-xs font-bold text-slate-800 mb-1.5">
              <span>Test Size</span>
              <span className="text-blue-600 font-bold">{Math.round(testSize * 100)}%</span>
            </div>
            <input
              type="range"
              min="0.1"
              max="0.5"
              step="0.05"
              value={testSize}
              onChange={(e) => setTestSize(parseFloat(e.target.value))}
              className="w-full h-1.5 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-blue-600"
            />
            <div className="flex justify-between text-[10px] text-slate-400 mt-1">
              <span>10%</span>
              <span>20% (Standard)</span>
              <span>30%</span>
              <span>50%</span>
            </div>
          </div>

          {/* Random State */}
          <div>
            <label className="block text-xs font-bold text-slate-800 mb-1.5">
              Random State
            </label>
            <input
              type="number"
              value={randomState}
              onChange={(e) => setRandomState(parseInt(e.target.value) || 42)}
              className="w-32 bg-slate-50 border border-slate-200 rounded-xl px-3 py-2 text-xs font-medium text-slate-800 focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>

          {/* Advanced Options Checkbox */}
          <div className="pt-2 border-t border-slate-100">
            <label className="flex items-center gap-2 text-xs font-medium text-slate-700 cursor-pointer">
              <input
                type="checkbox"
                checked={showAdvanced}
                onChange={(e) => setShowAdvanced(e.target.checked)}
                className="w-4 h-4 rounded text-blue-600 focus:ring-blue-500 border-slate-300"
              />
              <span>Show advanced options</span>
            </label>

            {showAdvanced && (
              <div className="mt-3 p-3 bg-slate-50 border border-slate-200 rounded-xl space-y-2 text-xs">
                <label className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={stratify}
                    onChange={(e) => setStratify(e.target.checked)}
                    className="w-3.5 h-3.5 rounded text-blue-600"
                  />
                  <span>Stratified split (preserves target class distribution across splits)</span>
                </label>
                <div className="text-[11px] text-slate-500">
                  Data leakage prevention: Transformer parameters (means, scales, vocabularies) are calculated purely on X_train.
                </div>
              </div>
            )}
          </div>

          {/* Apply button */}
          <div className="pt-4 flex justify-end">
            <button
              type="button"
              disabled={isApplying}
              onClick={handleApply}
              className="flex items-center gap-2 px-6 py-2.5 bg-blue-600 hover:bg-blue-700 disabled:opacity-50 text-white font-semibold text-xs rounded-xl shadow transition"
            >
              {isApplying ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  Fitting Pipeline...
                </>
              ) : (
                <>
                  Apply Preprocessing
                  <ArrowRight className="w-4 h-4" />
                </>
              )}
            </button>
          </div>
        </div>

        {/* Right Card: Detected Features matching screenshot */}
        <div className="bg-white rounded-2xl border border-slate-200/80 p-6 shadow-xs space-y-5 h-fit">
          <h3 className="text-xs font-bold text-slate-900 uppercase tracking-wider pb-2 border-b border-slate-100">
            Detected Features
          </h3>

          <div>
            <div className="flex items-center justify-between text-xs font-semibold text-slate-800 mb-1">
              <span>Numerical Features</span>
              <span className="px-2 py-0.5 bg-blue-50 text-blue-700 rounded-full text-[10px]">
                {numCols.length}
              </span>
            </div>
            <p className="text-[11px] text-slate-500 font-mono line-clamp-3">
              {numCols.join(', ') || 'None'}
            </p>
          </div>

          <div>
            <div className="flex items-center justify-between text-xs font-semibold text-slate-800 mb-1">
              <span>Categorical Features</span>
              <span className="px-2 py-0.5 bg-purple-50 text-purple-700 rounded-full text-[10px]">
                {catCols.length}
              </span>
            </div>
            <p className="text-[11px] text-slate-500 font-mono line-clamp-3">
              {catCols.join(', ') || 'None'}
            </p>
          </div>

          <div className="pt-2 border-t border-slate-100">
            <div className="text-[11px] font-semibold text-slate-500 mb-0.5">Target Column</div>
            <div className="text-xs font-bold text-slate-900">
              {targetColumn}
            </div>
          </div>

          <div>
            <div className="text-[11px] font-semibold text-slate-500 mb-1">Problem Type</div>
            <div className="inline-block px-3 py-1 bg-emerald-100 text-emerald-800 font-bold text-xs rounded-lg uppercase tracking-wider">
              {problemDetection?.detected_problem_type || 'Classification'}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
