import React, { useState } from 'react';
import { useStudio } from '../../context/StudioContext';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  Cell,
} from 'recharts';
import { Sparkles, ArrowRight, Info, Layers } from 'lucide-react';

export const ExplainabilityScreen: React.FC = () => {
  const {
    shapData,
    selectedModelName,
    selectedModelId,
    comparison,
    setSelectedModelId,
    setActiveStage,
    markStageCompleted,
  } = useStudio();

  const [activeTab, setActiveTab] = useState<'shap_summary' | 'shap_bar' | 'feature_importance'>('shap_summary');

  if (!shapData) {
    return (
      <div className="p-8 text-center text-slate-500">
        Loading SHAP explainability data...
      </div>
    );
  }

  const { explainer_type, feature_importance, shap_bar_data, scatter_samples, base_value } = shapData;

  // Group scatter points by feature for beeswarm visual
  const top10Features = feature_importance.slice(0, 10).map((f) => f.feature);

  return (
    <div className="max-w-6xl mx-auto py-8 px-6 space-y-6">
      {/* Header matching screenshot */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h2 className="text-xl font-bold text-slate-900 tracking-tight">
            Model Explainability - {selectedModelName}
          </h2>
          <p className="text-xs text-slate-500 mt-0.5">
            Explainer: <span className="font-semibold text-blue-600">{explainer_type}</span> • Base Value: {base_value}
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
              markStageCompleted('predict');
              setActiveStage('predict');
            }}
            className="flex items-center gap-2 px-5 py-2 bg-blue-600 hover:bg-blue-700 text-white font-semibold text-xs rounded-xl shadow transition"
          >
            Individual Prediction & Export
            <ArrowRight className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Tabs matching screenshot */}
      <div className="flex border-b border-slate-200 overflow-x-auto">
        <button
          onClick={() => setActiveTab('shap_summary')}
          className={`px-5 py-2.5 text-xs font-bold border-b-2 whitespace-nowrap transition-all ${
            activeTab === 'shap_summary'
              ? 'border-blue-600 text-blue-600'
              : 'border-transparent text-slate-500 hover:text-slate-700'
          }`}
        >
          SHAP Summary Plot
        </button>
        <button
          onClick={() => setActiveTab('shap_bar')}
          className={`px-5 py-2.5 text-xs font-bold border-b-2 whitespace-nowrap transition-all ${
            activeTab === 'shap_bar'
              ? 'border-blue-600 text-blue-600'
              : 'border-transparent text-slate-500 hover:text-slate-700'
          }`}
        >
          SHAP Bar Plot
        </button>
        <button
          onClick={() => setActiveTab('feature_importance')}
          className={`px-5 py-2.5 text-xs font-bold border-b-2 whitespace-nowrap transition-all ${
            activeTab === 'feature_importance'
              ? 'border-blue-600 text-blue-600'
              : 'border-transparent text-slate-500 hover:text-slate-700'
          }`}
        >
          Feature Importance
        </button>
      </div>

      {/* 1. SHAP Summary Plot (Beeswarm view matching reference image) */}
      {activeTab === 'shap_summary' && (
        <div className="bg-white rounded-2xl border border-slate-200/80 p-6 shadow-xs space-y-4">
          <div className="flex items-center justify-between pb-2 border-b border-slate-100">
            <div>
              <h3 className="text-xs font-bold text-slate-800 uppercase tracking-wider">
                SHAP Summary Plot (Feature Impact on Model Output)
              </h3>
              <p className="text-[11px] text-slate-400 mt-0.5">
                Horizontal dispersion shows SHAP value impact. Colors indicate feature value (Red = High, Blue = Low).
              </p>
            </div>

            {/* Color scale indicator matching screenshot */}
            <div className="flex items-center gap-2 text-xs">
              <span className="text-[10px] text-blue-600 font-bold">Low</span>
              <div className="w-24 h-2.5 rounded-full bg-gradient-to-r from-blue-500 via-purple-500 to-rose-500"></div>
              <span className="text-[10px] text-rose-600 font-bold">High</span>
              <span className="text-[10px] text-slate-400 ml-1">Feature value</span>
            </div>
          </div>

          {/* Beeswarm / Scatter Canvas visualization */}
          <div className="space-y-3 py-2">
            {top10Features.map((feat) => {
              const points = scatter_samples.filter((p) => p.feature === feat);
              return (
                <div key={feat} className="flex items-center gap-4 text-xs">
                  <div className="w-36 text-right font-medium text-slate-700 truncate text-[11px]" title={feat}>
                    {feat}
                  </div>

                  <div className="flex-1 relative h-8 bg-slate-50 rounded-lg border border-slate-200/60 overflow-hidden flex items-center">
                    {/* Zero reference vertical line */}
                    <div className="absolute left-1/2 top-0 bottom-0 w-px bg-slate-300 z-0"></div>

                    {points.map((pt, pIdx) => {
                      // Map shap_value (-2 to +2 typically) to percentage 0-100
                      const leftPct = Math.max(5, Math.min(95, 50 + pt.shap_value * 30));
                      // Feature value color interpolation (0 = blue, 1 = red)
                      const r = Math.round(59 + pt.feature_value * 180);
                      const g = Math.round(130 - pt.feature_value * 90);
                      const b = Math.round(246 - pt.feature_value * 180);
                      const color = `rgb(${r},${g},${b})`;

                      return (
                        <div
                          key={pIdx}
                          title={`Feature: ${feat}\nSHAP: ${pt.shap_value}\nNormalized Val: ${pt.feature_value}`}
                          className="absolute w-2.5 h-2.5 rounded-full opacity-75 hover:opacity-100 hover:scale-125 transition-transform"
                          style={{
                            left: `${leftPct}%`,
                            backgroundColor: color,
                            top: `${20 + (pIdx % 3) * 18}%`,
                          }}
                        />
                      );
                    })}
                  </div>
                </div>
              );
            })}

            <div className="flex justify-between text-[10px] text-slate-400 pt-2 px-36">
              <span>-1.0 (Decreases output)</span>
              <span>0.0 (No impact)</span>
              <span>+1.0 (Increases output)</span>
            </div>
          </div>
        </div>
      )}

      {/* 2. SHAP Bar Plot Tab */}
      {activeTab === 'shap_bar' && (
        <div className="bg-white rounded-2xl border border-slate-200/80 p-6 shadow-xs space-y-4">
          <h3 className="text-xs font-bold text-slate-800 uppercase tracking-wider">
            Mean |SHAP Value| (Average Impact Magnitude)
          </h3>

          <div className="h-96 w-full pt-2">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={shap_bar_data}
                layout="vertical"
                margin={{ top: 10, right: 30, left: 100, bottom: 20 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis type="number" tick={{ fontSize: 11 }} />
                <YAxis dataKey="feature" type="category" tick={{ fontSize: 11 }} />
                <Tooltip formatter={(val: any) => [`${val}`, 'Mean |SHAP|']} />
                <Bar dataKey="importance" fill="#2563eb" radius={[0, 6, 6, 0]} name="Impact" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* 3. Feature Importance Tab */}
      {activeTab === 'feature_importance' && (
        <div className="bg-white rounded-2xl border border-slate-200/80 p-6 shadow-xs space-y-4">
          <h3 className="text-xs font-bold text-slate-800 uppercase tracking-wider">
            Ranked Global Feature Importance
          </h3>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {feature_importance.map((f, i) => {
              const maxImp = feature_importance[0]?.importance || 1;
              const pct = (f.importance / maxImp) * 100;

              return (
                <div key={f.feature} className="p-3 bg-slate-50 rounded-xl border border-slate-200/60 space-y-1.5">
                  <div className="flex items-center justify-between text-xs">
                    <span className="font-semibold text-slate-800 flex items-center gap-1.5">
                      <span className="text-[10px] text-slate-400 font-mono w-4">{i + 1}.</span>
                      {f.feature}
                    </span>
                    <span className="font-mono text-slate-600 font-bold">{f.importance}</span>
                  </div>

                  <div className="w-full bg-slate-200 h-1.5 rounded-full overflow-hidden">
                    <div
                      className="bg-blue-600 h-full rounded-full transition-all duration-500"
                      style={{ width: `${pct}%` }}
                    />
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
