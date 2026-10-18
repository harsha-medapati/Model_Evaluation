import React, { useState } from 'react';
import { useStudio } from '../../context/StudioContext';
import {
  LineChart,
  ResponsiveContainer,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  BarChart,
  Bar,
  ScatterChart,
  Scatter,
} from 'recharts';
import { ArrowRight, Layers, Award, Activity } from 'lucide-react';

export const DetailedEvaluationScreen: React.FC = () => {
  const {
    comparison,
    selectedModelId,
    setSelectedModelId,
    selectedModelName,
    detailedEvaluation,
    setActiveStage,
    markStageCompleted,
  } = useStudio();

  const [activeTab, setActiveTab] = useState<
    'confusion_matrix' | 'roc_curve' | 'pr_curve' | 'cross_validation' | 'metrics' | 'regression_plots'
  >('confusion_matrix');

  if (!detailedEvaluation) {
    return (
      <div className="p-8 text-center text-slate-500">
        Loading detailed model evaluation...
      </div>
    );
  }

  const {
    model_name,
    model_type,
    metrics,
    confusion_matrix,
    classification_report,
    roc_curve,
    pr_curve,
    actual_vs_predicted,
    residuals,
    cv_results,
    training_time_sec,
  } = detailedEvaluation;

  const isClassification = model_type === 'classification';

  // Format ROC data for Recharts
  const rocChartData =
    roc_curve?.fpr.map((fpr, i) => ({
      fpr,
      tpr: roc_curve.tpr[i],
      baseline: fpr,
    })) || [];

  // Format PR data
  const prChartData =
    pr_curve?.recall.map((rec, i) => ({
      recall: rec,
      precision: pr_curve.precision[i],
    })) || [];

  // Format CV data
  const cvChartData =
    cv_results?.folds?.map((score: number, idx: number) => ({
      fold: `Fold ${idx + 1}`,
      score,
    })) || [];

  return (
    <div className="max-w-6xl mx-auto py-8 px-6 space-y-6">
      {/* Header with Model Selector */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h2 className="text-xl font-bold text-slate-900 tracking-tight">
            {selectedModelName} - Detailed Evaluation
          </h2>
          <p className="text-xs text-slate-500 mt-0.5">
            Diagnostic performance curves, confusion matrices, and fold distributions
          </p>
        </div>

        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2">
            <span className="text-xs text-slate-500 font-medium">Model:</span>
            <select
              value={selectedModelId || ''}
              onChange={(e) => setSelectedModelId(e.target.value)}
              className="bg-white border border-slate-300 rounded-xl px-3 py-1.5 text-xs font-semibold text-slate-800 focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              {comparison?.models.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.model_name}
                </option>
              ))}
            </select>
          </div>

          <button
            onClick={() => {
              markStageCompleted('explain');
              setActiveStage('explain');
            }}
            className="flex items-center gap-2 px-5 py-2 bg-blue-600 hover:bg-blue-700 text-white font-semibold text-xs rounded-xl shadow transition"
          >
            SHAP Explainability
            <ArrowRight className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Tabs matching screenshot */}
      <div className="flex border-b border-slate-200 overflow-x-auto">
        <button
          onClick={() => setActiveTab('metrics')}
          className={`px-5 py-2.5 text-xs font-bold border-b-2 whitespace-nowrap transition-all ${
            activeTab === 'metrics'
              ? 'border-blue-600 text-blue-600'
              : 'border-transparent text-slate-500 hover:text-slate-700'
          }`}
        >
          Metrics
        </button>

        {isClassification ? (
          <>
            <button
              onClick={() => setActiveTab('confusion_matrix')}
              className={`px-5 py-2.5 text-xs font-bold border-b-2 whitespace-nowrap transition-all ${
                activeTab === 'confusion_matrix'
                  ? 'border-blue-600 text-blue-600'
                  : 'border-transparent text-slate-500 hover:text-slate-700'
              }`}
            >
              Confusion Matrix
            </button>
            <button
              onClick={() => setActiveTab('roc_curve')}
              className={`px-5 py-2.5 text-xs font-bold border-b-2 whitespace-nowrap transition-all ${
                activeTab === 'roc_curve'
                  ? 'border-blue-600 text-blue-600'
                  : 'border-transparent text-slate-500 hover:text-slate-700'
              }`}
            >
              ROC Curve
            </button>
            <button
              onClick={() => setActiveTab('pr_curve')}
              className={`px-5 py-2.5 text-xs font-bold border-b-2 whitespace-nowrap transition-all ${
                activeTab === 'pr_curve'
                  ? 'border-blue-600 text-blue-600'
                  : 'border-transparent text-slate-500 hover:text-slate-700'
              }`}
            >
              Precision-Recall
            </button>
          </>
        ) : (
          <button
            onClick={() => setActiveTab('regression_plots')}
            className={`px-5 py-2.5 text-xs font-bold border-b-2 whitespace-nowrap transition-all ${
              activeTab === 'regression_plots'
                ? 'border-blue-600 text-blue-600'
                : 'border-transparent text-slate-500 hover:text-slate-700'
            }`}
          >
            Actual vs Predicted & Residuals
          </button>
        )}

        <button
          onClick={() => setActiveTab('cross_validation')}
          className={`px-5 py-2.5 text-xs font-bold border-b-2 whitespace-nowrap transition-all ${
            activeTab === 'cross_validation'
              ? 'border-blue-600 text-blue-600'
              : 'border-transparent text-slate-500 hover:text-slate-700'
          }`}
        >
          Cross Validation
        </button>
      </div>

      {/* Tab Content */}
      {/* 1. Confusion Matrix Tab matching reference image */}
      {activeTab === 'confusion_matrix' && confusion_matrix && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Left: Interactive Heatmap */}
          <div className="bg-white rounded-2xl border border-slate-200/80 p-6 shadow-xs">
            <h3 className="text-xs font-bold text-slate-800 uppercase tracking-wider mb-4">
              Confusion Matrix Heatmap
            </h3>

            <div className="flex flex-col items-center justify-center p-4">
              <div className="text-[11px] font-bold text-slate-500 mb-2">PREDICTED CLASS</div>
              <div className="flex">
                <div className="writing-vertical -rotate-180 text-[11px] font-bold text-slate-500 flex items-center justify-center mr-3">
                  ACTUAL CLASS
                </div>

                <div className="grid grid-cols-2 gap-2 max-w-sm">
                  {confusion_matrix.labels.map((rowLabel, rIdx) =>
                    confusion_matrix.labels.map((colLabel, cIdx) => {
                      const count = confusion_matrix.matrix[rIdx]?.[cIdx] ?? 0;
                      const norm = confusion_matrix.normalized?.[rIdx]?.[cIdx] ?? 0;
                      const isCorrect = rIdx === cIdx;

                      return (
                        <div
                          key={`${rIdx}-${cIdx}`}
                          className={`w-32 h-28 rounded-xl flex flex-col items-center justify-center border transition-all ${
                            isCorrect
                              ? 'bg-blue-600 text-white border-blue-700 shadow-sm'
                              : 'bg-slate-100 text-slate-800 border-slate-200'
                          }`}
                        >
                          <span className="text-2xl font-bold font-mono">{count}</span>
                          <span className={`text-[11px] ${isCorrect ? 'text-blue-100' : 'text-slate-500'}`}>
                            {(norm * 100).toFixed(1)}%
                          </span>
                          <span className={`text-[9px] uppercase tracking-wider mt-1 ${isCorrect ? 'text-blue-200' : 'text-slate-400'}`}>
                            Act: {rowLabel} / Pred: {colLabel}
                          </span>
                        </div>
                      );
                    })
                  )}
                </div>
              </div>
            </div>
          </div>

          {/* Right: Classification Report Table */}
          <div className="bg-white rounded-2xl border border-slate-200/80 p-6 shadow-xs space-y-4">
            <h3 className="text-xs font-bold text-slate-800 uppercase tracking-wider">
              Classification Report
            </h3>

            {classification_report && (
              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs">
                  <thead className="bg-slate-50 border-b border-slate-200 text-slate-600 text-[10px] uppercase">
                    <tr>
                      <th className="px-3 py-2 font-bold">Class</th>
                      <th className="px-3 py-2 font-bold text-center">Precision</th>
                      <th className="px-3 py-2 font-bold text-center">Recall</th>
                      <th className="px-3 py-2 font-bold text-center">F1-Score</th>
                      <th className="px-3 py-2 font-bold text-center">Support</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100">
                    {Object.entries(classification_report).map(([key, val]: [string, any]) => {
                      if (key === 'accuracy') {
                        return (
                          <tr key={key} className="bg-slate-50 font-bold">
                            <td className="px-3 py-2 text-slate-900">Accuracy</td>
                            <td colSpan={2} className="px-3 py-2 text-center text-slate-400">-</td>
                            <td className="px-3 py-2 text-center text-blue-600 font-mono">
                              {(val * 100).toFixed(1)}%
                            </td>
                            <td className="px-3 py-2 text-center text-slate-400">-</td>
                          </tr>
                        );
                      }
                      return (
                        <tr key={key} className="hover:bg-slate-50/60">
                          <td className="px-3 py-2 font-semibold text-slate-800">{key}</td>
                          <td className="px-3 py-2 text-center font-mono text-slate-700">{val.precision}</td>
                          <td className="px-3 py-2 text-center font-mono text-slate-700">{val.recall}</td>
                          <td className="px-3 py-2 text-center font-mono text-slate-700">{val.f1_score}</td>
                          <td className="px-3 py-2 text-center text-slate-500">{val.support}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </div>
      )}

      {/* 2. ROC Curve Tab */}
      {activeTab === 'roc_curve' && (
        <div className="bg-white rounded-2xl border border-slate-200/80 p-6 shadow-xs space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-xs font-bold text-slate-800 uppercase tracking-wider">
              Receiver Operating Characteristic (ROC Curve)
            </h3>
            {roc_curve && (
              <span className="px-3 py-1 bg-blue-100 text-blue-800 font-bold text-xs rounded-lg">
                ROC-AUC: {(roc_curve.auc * 100).toFixed(2)}%
              </span>
            )}
          </div>

          <div className="h-80 w-full pt-4">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={rocChartData} margin={{ top: 10, right: 30, left: 10, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis
                  dataKey="fpr"
                  label={{ value: 'False Positive Rate (1 - Specificity)', position: 'insideBottom', offset: -10 }}
                  tick={{ fontSize: 11 }}
                />
                <YAxis
                  label={{ value: 'True Positive Rate (Recall)', angle: -90, position: 'insideLeft' }}
                  tick={{ fontSize: 11 }}
                  domain={[0, 1]}
                />
                <Tooltip />
                <Line
                  type="monotone"
                  dataKey="tpr"
                  stroke="#2563eb"
                  strokeWidth={2.5}
                  name="Model ROC"
                  dot={false}
                />
                <Line
                  type="linear"
                  dataKey="baseline"
                  stroke="#94a3b8"
                  strokeDasharray="4 4"
                  name="Random Chance"
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* 3. Precision-Recall Tab */}
      {activeTab === 'pr_curve' && (
        <div className="bg-white rounded-2xl border border-slate-200/80 p-6 shadow-xs space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-xs font-bold text-slate-800 uppercase tracking-wider">
              Precision-Recall Curve
            </h3>
            {pr_curve && (
              <span className="px-3 py-1 bg-purple-100 text-purple-800 font-bold text-xs rounded-lg">
                Average Precision: {(pr_curve.avg_precision * 100).toFixed(2)}%
              </span>
            )}
          </div>

          <div className="h-80 w-full pt-4">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={prChartData} margin={{ top: 10, right: 30, left: 10, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis
                  dataKey="recall"
                  label={{ value: 'Recall', position: 'insideBottom', offset: -10 }}
                  tick={{ fontSize: 11 }}
                  domain={[0, 1]}
                />
                <YAxis
                  label={{ value: 'Precision', angle: -90, position: 'insideLeft' }}
                  tick={{ fontSize: 11 }}
                  domain={[0, 1]}
                />
                <Tooltip />
                <Line
                  type="monotone"
                  dataKey="precision"
                  stroke="#7c3aed"
                  strokeWidth={2.5}
                  name="Precision"
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* 4. Cross Validation Tab */}
      {activeTab === 'cross_validation' && cv_results && (
        <div className="bg-white rounded-2xl border border-slate-200/80 p-6 shadow-xs space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-xs font-bold text-slate-800 uppercase tracking-wider">
                Cross-Validation Folds Performance ({cv_results.metric})
              </h3>
              <p className="text-xs text-slate-500 mt-0.5">
                Mean Score: <span className="font-bold text-slate-900">{cv_results.mean}</span> (± {cv_results.std} std dev)
              </p>
            </div>
          </div>

          <div className="h-72 w-full pt-4">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={cvChartData} margin={{ top: 10, right: 30, left: 10, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis dataKey="fold" tick={{ fontSize: 11 }} />
                <YAxis domain={[0, 1]} tick={{ fontSize: 11 }} />
                <Tooltip />
                <Bar dataKey="score" fill="#3b82f6" radius={[6, 6, 0, 0]} name="Score" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* 5. Metrics Tab */}
      {activeTab === 'metrics' && (
        <div className="bg-white rounded-2xl border border-slate-200/80 p-6 shadow-xs space-y-6">
          <h3 className="text-xs font-bold text-slate-800 uppercase tracking-wider">
            Comprehensive Evaluation Metrics
          </h3>

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
            {Object.entries(metrics).map(([k, v]) => (
              <div key={k} className="p-4 bg-slate-50 rounded-xl border border-slate-200/60">
                <div className="text-[11px] font-semibold text-slate-500 mb-1">{k}</div>
                <div className="text-xl font-bold text-slate-900">
                  {typeof v === 'number'
                    ? isClassification && v <= 1.0
                      ? `${(v * 100).toFixed(1)}%`
                      : v.toFixed(k.includes('R²') ? 4 : 2)
                    : String(v)}
                </div>
              </div>
            ))}
          </div>

          <div className="p-4 bg-slate-50 rounded-xl border border-slate-200/60 space-y-2">
            <div className="text-xs font-bold text-slate-800">Training Metadata</div>
            <div className="text-xs text-slate-600">Training Duration: <span className="font-semibold">{training_time_sec}s</span></div>
            <div className="text-xs text-slate-600">Model Algorithm: <span className="font-semibold">{model_name}</span></div>
          </div>
        </div>
      )}

      {/* 6. Regression Plots Tab */}
      {activeTab === 'regression_plots' && actual_vs_predicted && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="bg-white rounded-2xl border border-slate-200/80 p-6 shadow-xs space-y-4">
            <h3 className="text-xs font-bold text-slate-800 uppercase tracking-wider">
              Actual vs Predicted Values
            </h3>
            <div className="h-72 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <ScatterChart margin={{ top: 10, right: 30, left: 10, bottom: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                  <XAxis dataKey="actual" name="Actual" label={{ value: 'Actual', position: 'insideBottom', offset: -10 }} />
                  <YAxis dataKey="predicted" name="Predicted" label={{ value: 'Predicted', angle: -90, position: 'insideLeft' }} />
                  <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                  <Scatter data={actual_vs_predicted} fill="#2563eb" />
                </ScatterChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="bg-white rounded-2xl border border-slate-200/80 p-6 shadow-xs space-y-4">
            <h3 className="text-xs font-bold text-slate-800 uppercase tracking-wider">
              Residuals Distribution
            </h3>
            {residuals && (
              <div className="h-72 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={residuals.distribution} margin={{ top: 10, right: 30, left: 10, bottom: 20 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                    <XAxis dataKey="bin_center" tick={{ fontSize: 10 }} />
                    <YAxis tick={{ fontSize: 10 }} />
                    <Tooltip />
                    <Bar dataKey="count" fill="#8b5cf6" radius={[4, 4, 0, 0]} name="Residuals Count" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};
