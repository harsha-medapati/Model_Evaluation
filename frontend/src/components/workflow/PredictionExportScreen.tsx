import React, { useState, useEffect } from 'react';
import { useStudio } from '../../context/StudioContext';
import { apiClient } from '../../api/client';
import {
  CheckCircle2,
  AlertCircle,
  Download,
  FileSpreadsheet,
  FileText,
  Play,
  Loader2,
  Sparkles,
  Layers,
} from 'lucide-react';
import { PredictionResponse } from '../../types';

export const PredictionExportScreen: React.FC = () => {
  const {
    selectedModelId,
    selectedModelName,
    experimentSummary,
    predictionSchema,
    comparison,
    setSelectedModelId,
  } = useStudio();

  const [formValues, setFormValues] = useState<Record<string, any>>({});
  const [predictionResult, setPredictionResult] = useState<PredictionResponse | null>(null);
  const [isPredicting, setIsPredicting] = useState<boolean>(false);
  const [predictError, setPredictError] = useState<string | null>(null);

  // Initialize form values from schema defaults
  useEffect(() => {
    if (predictionSchema?.fields) {
      const initial: Record<string, any> = {};
      predictionSchema.fields.forEach((f) => {
        initial[f.name] = f.default_value !== undefined ? f.default_value : '';
      });
      setFormValues(initial);
    }
  }, [predictionSchema]);

  const handleInputChange = (field: string, value: any) => {
    setFormValues((prev) => ({ ...prev, [field]: value }));
  };

  const handlePredict = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedModelId) return;
    setIsPredicting(true);
    setPredictError(null);
    try {
      const res = await apiClient.makePrediction(selectedModelId, formValues);
      setPredictionResult(res);
    } catch (err: any) {
      setPredictError(err.response?.data?.detail || 'Failed to make prediction.');
    } finally {
      setIsPredicting(false);
    }
  };

  return (
    <div className="max-w-6xl mx-auto py-8 px-6 space-y-8">
      {/* Title & Model Selector */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h2 className="text-xl font-bold text-slate-900 tracking-tight">
            Individual Prediction & Export
          </h2>
          <p className="text-xs text-slate-500 mt-0.5">
            Test single inputs, inspect real-time SHAP feature contributions, and export model packages
          </p>
        </div>

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
      </div>

      {/* Main Grid: Make Prediction Form (Left) & Prediction Result (Right) */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Left: Make a Prediction Form matching reference image */}
        <div className="bg-white rounded-2xl border border-slate-200/80 p-6 shadow-xs space-y-4">
          <h3 className="text-xs font-bold text-slate-800 uppercase tracking-wider pb-2 border-b border-slate-100 flex items-center gap-2">
            <Play className="w-3.5 h-3.5 text-blue-600" />
            Make a Prediction
          </h3>

          {predictError && (
            <div className="p-3 bg-red-50 border border-red-200 text-red-700 text-xs rounded-xl">
              {predictError}
            </div>
          )}

          <form onSubmit={handlePredict} className="space-y-3">
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 max-h-[380px] overflow-y-auto p-1">
              {predictionSchema?.fields.map((field) => (
                <div key={field.name}>
                  <label className="block text-[11px] font-semibold text-slate-700 mb-1">
                    {field.label}
                  </label>

                  {field.data_type === 'categorical' && field.options ? (
                    <select
                      value={formValues[field.name] ?? ''}
                      onChange={(e) => handleInputChange(field.name, e.target.value)}
                      className="w-full bg-slate-50 border border-slate-200 rounded-xl px-2.5 py-1.5 text-xs font-medium text-slate-800 focus:outline-none focus:ring-2 focus:ring-blue-500"
                    >
                      {field.options.map((opt) => (
                        <option key={opt} value={opt}>
                          {opt}
                        </option>
                      ))}
                    </select>
                  ) : (
                    <input
                      type="number"
                      step={field.step || 1}
                      value={formValues[field.name] ?? ''}
                      onChange={(e) => handleInputChange(field.name, parseFloat(e.target.value) || 0)}
                      className="w-full bg-slate-50 border border-slate-200 rounded-xl px-2.5 py-1.5 text-xs font-medium text-slate-800 focus:outline-none focus:ring-2 focus:ring-blue-500"
                    />
                  )}
                </div>
              ))}
            </div>

            <div className="pt-2">
              <button
                type="submit"
                disabled={isPredicting || !selectedModelId}
                className="w-full flex items-center justify-center gap-2 py-2.5 bg-blue-600 hover:bg-blue-700 disabled:opacity-50 text-white font-bold text-xs rounded-xl shadow transition"
              >
                {isPredicting ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin" />
                    Calculating Prediction & SHAP...
                  </>
                ) : (
                  <>
                    Predict
                  </>
                )}
              </button>
            </div>
          </form>
        </div>

        {/* Right: Prediction Result & Feature Contributions matching reference image */}
        <div className="bg-white rounded-2xl border border-slate-200/80 p-6 shadow-xs space-y-5">
          <h3 className="text-xs font-bold text-slate-800 uppercase tracking-wider pb-2 border-b border-slate-100 flex items-center gap-2">
            <Sparkles className="w-3.5 h-3.5 text-emerald-600" />
            Prediction Result
          </h3>

          {predictionResult ? (
            <div className="space-y-5">
              {/* Prediction Badge matching reference UI */}
              <div className="p-4 bg-emerald-50/70 border border-emerald-200 rounded-2xl flex items-center gap-4">
                <div className="w-12 h-12 rounded-xl bg-emerald-500 text-white flex items-center justify-center font-bold text-lg shadow-sm">
                  <CheckCircle2 className="w-7 h-7" />
                </div>
                <div>
                  <div className="text-[11px] font-bold text-emerald-800 uppercase tracking-wider">
                    Model Decision
                  </div>
                  <div className="text-xl font-bold text-slate-900 leading-tight">
                    {String(predictionResult.prediction)}
                  </div>
                  {predictionResult.probability !== null && predictionResult.probability !== undefined && (
                    <div className="text-xs font-semibold text-emerald-700 mt-0.5">
                      Probability: {(predictionResult.probability * 100).toFixed(1)}%
                    </div>
                  )}
                </div>
              </div>

              {/* Feature Contribution (SHAP) list matching reference UI */}
              <div>
                <h4 className="text-xs font-bold text-slate-800 uppercase tracking-wider mb-2">
                  Feature Contribution (SHAP)
                </h4>
                <div className="space-y-2">
                  {predictionResult.feature_contributions.map((fc) => {
                    const isPos = fc.contribution >= 0;
                    return (
                      <div
                        key={fc.feature}
                        className="flex items-center justify-between p-2.5 bg-slate-50/80 border border-slate-200/60 rounded-xl text-xs"
                      >
                        <div className="flex items-center gap-2">
                          <span className="font-semibold text-slate-800">{fc.feature}</span>
                          {fc.value !== undefined && fc.value !== '' && (
                            <span className="text-[10px] text-slate-400">({String(fc.value)})</span>
                          )}
                        </div>

                        <div
                          className={`font-mono font-bold text-xs px-2 py-0.5 rounded ${
                            isPos
                              ? 'bg-emerald-100 text-emerald-800'
                              : 'bg-rose-100 text-rose-800'
                          }`}
                        >
                          {isPos ? `+${fc.contribution.toFixed(2)}` : fc.contribution.toFixed(2)}
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>
          ) : (
            <div className="p-12 text-center text-slate-400 text-xs border border-dashed border-slate-200 rounded-xl">
              Fill in feature values on the left and click "Predict" to see the model output and real-time SHAP impact waterfall.
            </div>
          )}
        </div>
      </div>

      {/* Export Section matching reference image bottom */}
      <div className="bg-white rounded-2xl border border-slate-200/80 p-6 shadow-xs space-y-4">
        <h3 className="text-xs font-bold text-slate-800 uppercase tracking-wider pb-2 border-b border-slate-100">
          Export Results & Artifacts
        </h3>

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          {/* Download Model Package */}
          <a
            href={selectedModelId ? apiClient.getModelExportUrl(selectedModelId) : '#'}
            target="_blank"
            rel="noreferrer"
            className="flex items-center justify-center gap-2.5 p-3.5 bg-blue-600 hover:bg-blue-700 text-white rounded-xl text-xs font-bold shadow-md transition text-center"
          >
            <Download className="w-4 h-4" />
            Download Model (.zip)
          </a>

          {/* Download Comparison CSV */}
          <a
            href={experimentSummary ? apiClient.getComparisonCsvUrl(experimentSummary.experiment_id) : '#'}
            target="_blank"
            rel="noreferrer"
            className="flex items-center justify-center gap-2.5 p-3.5 bg-emerald-600 hover:bg-emerald-700 text-white rounded-xl text-xs font-bold shadow-md transition text-center"
          >
            <FileSpreadsheet className="w-4 h-4" />
            Download Comparison (.csv)
          </a>

          {/* Generate PDF Report */}
          <a
            href={experimentSummary ? apiClient.getPdfReportUrl(experimentSummary.experiment_id) : '#'}
            target="_blank"
            rel="noreferrer"
            className="flex items-center justify-center gap-2.5 p-3.5 bg-rose-600 hover:bg-rose-700 text-white rounded-xl text-xs font-bold shadow-md transition text-center"
          >
            <FileText className="w-4 h-4" />
            Generate Report (.pdf)
          </a>
        </div>
      </div>
    </div>
  );
};
