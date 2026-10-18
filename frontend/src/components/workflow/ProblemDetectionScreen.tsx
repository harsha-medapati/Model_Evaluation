import React, { useState } from 'react';
import { useStudio } from '../../context/StudioContext';
import {
  CheckCircle2,
  Info,
  ArrowRight,
  TrendingUp,
  BarChart,
} from 'lucide-react';
import { ResponsiveContainer, BarChart as RechartsBarChart, Bar, XAxis, YAxis, Tooltip, Cell } from 'recharts';

export const ProblemDetectionScreen: React.FC = () => {
  const {
    problemDetection,
    targetColumn,
    setActiveStage,
    markStageCompleted,
  } = useStudio();

  const [overrideType, setOverrideType] = useState<string>(
    problemDetection?.detected_problem_type || 'classification'
  );

  if (!problemDetection) {
    return (
      <div className="p-8 text-center text-slate-500">
        Please configure preprocessing first.
      </div>
    );
  }

  const { detected_problem_type, reason, classes_count, class_distribution, target_stats } = problemDetection;

  const handleNext = () => {
    markStageCompleted('problem');
    markStageCompleted('train');
    setActiveStage('train');
  };

  const chartColors = ['#2563eb', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6'];

  return (
    <div className="max-w-4xl mx-auto py-8 px-6 space-y-6">
      <div>
        <h2 className="text-xl font-bold text-slate-900 tracking-tight">Detected Problem Type</h2>
        <p className="text-xs text-slate-500 mt-0.5">
          Automatic problem detection derived from target column characteristics
        </p>
      </div>

      {/* Main Detection Card matching reference UI */}
      <div className="bg-white rounded-2xl border border-slate-200/80 p-8 shadow-xs space-y-6">
        {/* Banner with big checkmark */}
        <div className="flex items-start gap-4 p-5 bg-emerald-50/60 border border-emerald-200/80 rounded-2xl">
          <div className="w-12 h-12 rounded-full bg-emerald-500 text-white flex items-center justify-center shrink-0 shadow-sm">
            <CheckCircle2 className="w-7 h-7" />
          </div>
          <div>
            <span className="text-[11px] font-bold text-emerald-700 uppercase tracking-wider">
              Automatic Detection
            </span>
            <h3 className="text-2xl font-bold text-slate-900 capitalize leading-tight">
              {detected_problem_type}
            </h3>
            <p className="text-xs text-slate-600 mt-1">{reason}</p>
          </div>
        </div>

        {/* Target Details Grid */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 pt-2">
          <div className="p-4 bg-slate-50 rounded-xl border border-slate-200/60">
            <div className="text-[11px] font-semibold text-slate-500 mb-1">Target Column</div>
            <div className="text-sm font-bold text-slate-900 font-mono">{targetColumn}</div>
          </div>

          {detected_problem_type === 'classification' && (
            <div className="p-4 bg-slate-50 rounded-xl border border-slate-200/60">
              <div className="text-[11px] font-semibold text-slate-500 mb-1">Number of Classes</div>
              <div className="text-sm font-bold text-slate-900">{classes_count} distinct classes</div>
            </div>
          )}

          {detected_problem_type === 'regression' && target_stats && (
            <div className="p-4 bg-slate-50 rounded-xl border border-slate-200/60">
              <div className="text-[11px] font-semibold text-slate-500 mb-1">Target Mean & Std</div>
              <div className="text-sm font-bold text-slate-900">
                {target_stats.mean.toLocaleString()} ± {target_stats.std.toLocaleString()}
              </div>
            </div>
          )}

          <div className="p-4 bg-slate-50 rounded-xl border border-slate-200/60">
            <div className="text-[11px] font-semibold text-slate-500 mb-1">Evaluation Metrics</div>
            <div className="text-xs font-semibold text-blue-600">
              {problemDetection.recommended_metrics.slice(0, 3).join(', ')}
            </div>
          </div>
        </div>

        {/* Target Distribution Chart */}
        {detected_problem_type === 'classification' && class_distribution && (
          <div className="pt-2">
            <h4 className="text-xs font-bold text-slate-800 uppercase tracking-wider mb-3">
              Class Distribution
            </h4>
            <div className="h-44 w-full bg-slate-50 rounded-xl p-3 border border-slate-200/60">
              <ResponsiveContainer width="100%" height="100%">
                <RechartsBarChart data={class_distribution} margin={{ top: 10, right: 20, left: 0, bottom: 20 }}>
                  <XAxis dataKey="category" tick={{ fontSize: 11 }} />
                  <YAxis tick={{ fontSize: 11 }} unit="%" />
                  <Tooltip
                    formatter={(val: any, _name: any, item: any) => [
                      `${val}% (${item.payload.count} samples)`,
                      'Percentage',
                    ]}
                  />
                  <Bar dataKey="percentage" radius={[6, 6, 0, 0]}>
                    {class_distribution.map((_entry, index) => (
                      <Cell key={`cell-${index}`} fill={chartColors[index % chartColors.length]} />
                    ))}
                  </Bar>
                </RechartsBarChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}

        {/* Manual Override dropdown matching screenshot */}
        <div className="p-4 bg-blue-50/50 border border-blue-200/70 rounded-xl space-y-2">
          <div className="flex items-center gap-2 text-xs font-semibold text-blue-900">
            <Info className="w-4 h-4 text-blue-600" />
            <span>You can manually override the detected problem type if needed.</span>
          </div>

          <div className="max-w-xs pt-1">
            <select
              value={overrideType}
              onChange={(e) => setOverrideType(e.target.value)}
              className="w-full bg-white border border-blue-300 rounded-xl px-3 py-2 text-xs font-medium text-slate-800 focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="classification">
                Classification {detected_problem_type === 'classification' ? '(Auto-detected)' : ''}
              </option>
              <option value="regression">
                Regression {detected_problem_type === 'regression' ? '(Auto-detected)' : ''}
              </option>
            </select>
          </div>
        </div>

        {/* Bottom CTA */}
        <div className="flex justify-end pt-2">
          <button
            type="button"
            onClick={handleNext}
            className="flex items-center gap-2 px-6 py-2.5 bg-blue-600 hover:bg-blue-700 text-white font-semibold text-xs rounded-xl shadow transition"
          >
            Select Models to Train
            <ArrowRight className="w-4 h-4" />
          </button>
        </div>
      </div>
    </div>
  );
};
