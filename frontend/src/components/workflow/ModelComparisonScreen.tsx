import React, { useState } from 'react';
import { useStudio } from '../../context/StudioContext';
import { apiClient } from '../../api/client';
import {
  Trophy,
  ArrowRight,
  TrendingUp,
  Clock,
  Sparkles,
  BarChart2,
  ChevronDown,
} from 'lucide-react';

export const ModelComparisonScreen: React.FC = () => {
  const {
    comparison,
    setComparison,
    experimentSummary,
    selectedModelId,
    setSelectedModelId,
    setActiveStage,
    markStageCompleted,
  } = useStudio();

  const [sortMetric, setSortMetric] = useState<string>('F1 Score');

  if (!comparison || comparison.models.length === 0) {
    return (
      <div className="p-8 text-center text-slate-500">
        No completed models available for comparison. Please train models first.
      </div>
    );
  }

  const { problem_type, models, recommended_model, metric_keys } = comparison;

  const handleSortChange = async (metric: string) => {
    setSortMetric(metric);
    if (experimentSummary) {
      const updated = await apiClient.compareModels(experimentSummary.experiment_id, metric);
      setComparison(updated);
    }
  };

  const handleSelectModel = (modelId: string) => {
    setSelectedModelId(modelId);
  };

  const handleProceedToEvaluation = () => {
    markStageCompleted('evaluate');
    setActiveStage('evaluate');
  };

  return (
    <div className="max-w-6xl mx-auto py-8 px-6 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-bold text-slate-900 tracking-tight">
            Model Performance Comparison ({problem_type === 'classification' ? 'Classification' : 'Regression'})
          </h2>
          <p className="text-xs text-slate-500 mt-0.5">
            Consistent evaluation across all trained algorithms on identical test folds
          </p>
        </div>

        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2 text-xs">
            <span className="text-slate-500 font-medium">Rank by:</span>
            <select
              value={sortMetric}
              onChange={(e) => handleSortChange(e.target.value)}
              className="bg-white border border-slate-300 rounded-lg px-2.5 py-1.5 text-xs font-semibold text-slate-800 focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              {metric_keys.map((k) => (
                <option key={k} value={k}>
                  {k}
                </option>
              ))}
            </select>
          </div>

          <button
            onClick={handleProceedToEvaluation}
            className="flex items-center gap-2 px-5 py-2 bg-blue-600 hover:bg-blue-700 text-white font-semibold text-xs rounded-xl shadow transition"
          >
            Detailed Results
            <ArrowRight className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Comparison Table matching reference image */}
      <div className="bg-white rounded-2xl border border-slate-200/80 shadow-xs overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="bg-slate-50/80 border-b border-slate-200 text-slate-600 uppercase text-[10px]">
              <tr>
                <th className="px-4 py-3 font-bold w-12 text-center">Rank</th>
                <th className="px-4 py-3 font-bold">Model</th>
                {metric_keys.map((mk) => (
                  <th key={mk} className="px-4 py-3 font-bold text-center">
                    {mk}
                  </th>
                ))}
                <th className="px-4 py-3 font-bold text-center">Train Time</th>
                <th className="px-4 py-3 font-bold text-center">Action</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {models.map((m) => {
                const isSelected = m.id === selectedModelId;
                const isChampion = m.id === recommended_model?.model_id;

                return (
                  <tr
                    key={m.id}
                    onClick={() => handleSelectModel(m.id)}
                    className={`cursor-pointer transition-colors ${
                      isSelected
                        ? 'bg-blue-50/70 font-semibold'
                        : 'hover:bg-slate-50/80'
                    }`}
                  >
                    <td className="px-4 py-3 text-center">
                      {isChampion ? (
                        <span className="w-5 h-5 rounded-full bg-amber-100 text-amber-700 flex items-center justify-center font-bold text-[10px] mx-auto">
                          1
                        </span>
                      ) : (
                        <span className="text-slate-400 text-xs">{m.rank}</span>
                      )}
                    </td>

                    <td className="px-4 py-3">
                      <div className="flex items-center gap-2">
                        <span className="text-slate-900 font-semibold">{m.model_name}</span>
                        {isChampion && (
                          <span className="px-1.5 py-0.5 bg-amber-100 text-amber-800 rounded text-[9px] font-bold">
                            Champion
                          </span>
                        )}
                      </div>
                    </td>

                    {metric_keys.map((mk) => {
                      const val = m.metrics[mk];
                      let formatted = '-';
                      if (typeof val === 'number') {
                        formatted =
                          problem_type === 'classification' && val <= 1.0
                            ? `${(val * 100).toFixed(1)}%`
                            : val.toFixed(mk.includes('R²') ? 4 : 2);
                      }

                      return (
                        <td key={mk} className="px-4 py-3 text-center text-slate-800">
                          {formatted}
                        </td>
                      );
                    })}

                    <td className="px-4 py-3 text-center">
                      <span className="px-2 py-0.5 bg-emerald-50 text-emerald-700 rounded-md font-mono text-[11px] font-medium">
                        {m.training_time_sec.toFixed(1)}s
                      </span>
                    </td>

                    <td className="px-4 py-3 text-center">
                      <button
                        type="button"
                        onClick={(e) => {
                          e.stopPropagation();
                          handleSelectModel(m.id);
                          handleProceedToEvaluation();
                        }}
                        className={`px-3 py-1 rounded-lg text-[11px] font-semibold transition ${
                          isSelected
                            ? 'bg-blue-600 text-white'
                            : 'bg-slate-100 hover:bg-slate-200 text-slate-700'
                        }`}
                      >
                        Inspect
                      </button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Recommended Model Card matching reference UI */}
      {recommended_model && (
        <div className="bg-gradient-to-r from-amber-50/80 via-white to-amber-50/40 rounded-2xl border border-amber-200/80 p-6 shadow-xs flex items-center gap-5">
          <div className="w-14 h-14 rounded-2xl bg-amber-500/10 border border-amber-300 flex items-center justify-center text-amber-500 shrink-0">
            <Trophy className="w-8 h-8 text-amber-600" />
          </div>

          <div className="space-y-1">
            <div className="text-[11px] font-bold text-amber-800 uppercase tracking-wider flex items-center gap-1.5">
              <Sparkles className="w-3.5 h-3.5" />
              Recommended Model
            </div>
            <h3 className="text-xl font-bold text-slate-900">
              {recommended_model.model_name}
            </h3>
            <p className="text-xs text-slate-600 leading-relaxed">
              {recommended_model.rationale}
            </p>
          </div>
        </div>
      )}
    </div>
  );
};
