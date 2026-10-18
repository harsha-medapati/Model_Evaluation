import React from 'react';
import { useStudio } from '../../context/StudioContext';
import {
  FileText,
  Hash,
  AlertTriangle,
  Copy,
  Binary,
  Tag,
  Target,
  KeyRound,
  ArrowRight,
  CheckCircle2,
} from 'lucide-react';

export const DataProfilingScreen: React.FC = () => {
  const { profile, targetColumn, setActiveStage, markStageCompleted } = useStudio();

  if (!profile) {
    return (
      <div className="p-8 text-center text-slate-500">
        No dataset profile available. Please upload a dataset first.
      </div>
    );
  }

  const { overview, columns, quality, data_preview, preview_columns } = profile;

  const handleNext = () => {
    markStageCompleted('profile');
    markStageCompleted('preprocess');
    setActiveStage('preprocess');
  };

  const statCards = [
    { label: 'Rows', value: overview.row_count.toLocaleString(), icon: <FileText className="w-4 h-4 text-blue-600" />, bg: 'bg-blue-50' },
    { label: 'Columns', value: overview.col_count, icon: <Hash className="w-4 h-4 text-indigo-600" />, bg: 'bg-indigo-50' },
    { label: 'Missing Values', value: overview.missing_values_count, icon: <AlertTriangle className="w-4 h-4 text-amber-600" />, bg: 'bg-amber-50' },
    { label: 'Duplicate Rows', value: overview.duplicate_rows_count, icon: <Copy className="w-4 h-4 text-rose-600" />, bg: 'bg-rose-50' },
    { label: 'Numerical Features', value: overview.numerical_features_count, icon: <Binary className="w-4 h-4 text-emerald-600" />, bg: 'bg-emerald-50' },
    { label: 'Categorical Features', value: overview.categorical_features_count, icon: <Tag className="w-4 h-4 text-purple-600" />, bg: 'bg-purple-50' },
    { label: 'Target Column', value: overview.target_column || 1, icon: <Target className="w-4 h-4 text-cyan-600" />, bg: 'bg-cyan-50' },
    { label: 'ID Columns', value: overview.id_column ? 1 : 0, icon: <KeyRound className="w-4 h-4 text-slate-600" />, bg: 'bg-slate-100' },
  ];

  return (
    <div className="max-w-6xl mx-auto py-8 px-6 space-y-8">
      {/* Title */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-bold text-slate-900 tracking-tight">Data Profiling</h2>
          <p className="text-xs text-slate-500 mt-0.5">Automated structural, statistical, and quality analysis</p>
        </div>
        <button
          onClick={handleNext}
          className="flex items-center gap-2 px-5 py-2.5 bg-blue-600 hover:bg-blue-700 text-white font-semibold text-xs rounded-xl shadow transition"
        >
          Configure Preprocessing
          <ArrowRight className="w-4 h-4" />
        </button>
      </div>

      {/* Dataset Overview Stat Cards matching reference image */}
      <div>
        <h3 className="text-xs font-bold text-slate-700 uppercase tracking-wider mb-3">
          Dataset Overview
        </h3>
        <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-8 gap-3">
          {statCards.map((card, i) => (
            <div
              key={i}
              className="bg-white border border-slate-200/80 rounded-xl p-3 shadow-xs flex flex-col items-center text-center"
            >
              <div className={`w-8 h-8 rounded-lg ${card.bg} flex items-center justify-center mb-1.5`}>
                {card.icon}
              </div>
              <div className="text-base font-bold text-slate-900 leading-tight">
                {card.value}
              </div>
              <div className="text-[10px] text-slate-400 font-medium truncate w-full">
                {card.label}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Quality Warnings */}
      {quality.warnings.length > 0 && (
        <div className="p-4 bg-amber-50/70 border border-amber-200 rounded-xl text-xs text-amber-800 space-y-1">
          <div className="font-semibold flex items-center gap-1.5 text-amber-900">
            <AlertTriangle className="w-4 h-4 text-amber-600" />
            <span>Data Quality Insights</span>
          </div>
          <ul className="list-disc list-inside space-y-0.5 text-[11px] text-amber-800">
            {quality.warnings.map((w, idx) => (
              <li key={idx}>{w}</li>
            ))}
          </ul>
        </div>
      )}

      {/* Data Preview Table */}
      <div className="bg-white border border-slate-200/80 rounded-xl shadow-xs overflow-hidden">
        <div className="p-4 border-b border-slate-100 flex items-center justify-between">
          <h3 className="text-xs font-bold text-slate-800 uppercase tracking-wider">
            Data Preview (First 10 Rows)
          </h3>
          <span className="text-[11px] text-slate-400">
            Showing sample from {overview.row_count} total rows
          </span>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="bg-slate-50 border-b border-slate-200 text-slate-600 uppercase text-[10px]">
              <tr>
                {preview_columns.map((col) => (
                  <th key={col} className="px-3 py-2.5 font-semibold">
                    <div className="flex items-center gap-1">
                      <span>{col}</span>
                      {col === targetColumn && (
                        <span className="px-1.5 py-0.2 bg-blue-100 text-blue-700 rounded text-[9px] font-bold">
                          TARGET
                        </span>
                      )}
                    </div>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {data_preview.map((row, rowIdx) => (
                <tr key={rowIdx} className="hover:bg-slate-50/80 transition-colors">
                  {preview_columns.map((col) => {
                    const val = row[col];
                    const isTarget = col === targetColumn;
                    const isApproved = val === 'Approved' || val === 1 || val === true;
                    const isRejected = val === 'Rejected' || val === 0 || val === false;

                    return (
                      <td key={col} className="px-3 py-2 text-slate-700 whitespace-nowrap">
                        {isTarget && (isApproved || isRejected) ? (
                          <span
                            className={`inline-block px-2 py-0.5 rounded text-[11px] font-semibold ${
                              isApproved
                                ? 'bg-emerald-100 text-emerald-700'
                                : 'bg-rose-100 text-rose-700'
                            }`}
                          >
                            {String(val)}
                          </span>
                        ) : val === null || val === undefined ? (
                          <span className="text-slate-400 italic">null</span>
                        ) : (
                          String(val)
                        )}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Column Information Table */}
      <div className="bg-white border border-slate-200/80 rounded-xl shadow-xs overflow-hidden">
        <div className="p-4 border-b border-slate-100">
          <h3 className="text-xs font-bold text-slate-800 uppercase tracking-wider">
            Column Information & Statistics
          </h3>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="bg-slate-50 border-b border-slate-200 text-slate-600 uppercase text-[10px]">
              <tr>
                <th className="px-3 py-2.5 font-semibold">Column</th>
                <th className="px-3 py-2.5 font-semibold">Data Type</th>
                <th className="px-3 py-2.5 font-semibold">Missing %</th>
                <th className="px-3 py-2.5 font-semibold">Unique</th>
                <th className="px-3 py-2.5 font-semibold">Mean</th>
                <th className="px-3 py-2.5 font-semibold">Std</th>
                <th className="px-3 py-2.5 font-semibold">Min</th>
                <th className="px-3 py-2.5 font-semibold">Max</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {columns.map((c) => (
                <tr key={c.name} className="hover:bg-slate-50/80 transition-colors">
                  <td className="px-3 py-2 font-medium text-slate-900 flex items-center gap-1.5">
                    {c.name}
                    {c.name === targetColumn && (
                      <span className="px-1.5 py-0.5 bg-blue-100 text-blue-700 rounded text-[9px] font-bold">
                        Target
                      </span>
                    )}
                    {c.is_potential_id && (
                      <span className="px-1.5 py-0.5 bg-slate-100 text-slate-600 rounded text-[9px]">
                        ID
                      </span>
                    )}
                  </td>
                  <td className="px-3 py-2 text-slate-500 font-mono text-[11px]">{c.data_type}</td>
                  <td className="px-3 py-2">
                    <span
                      className={`inline-block px-1.5 py-0.5 rounded text-[10px] font-semibold ${
                        c.missing_percentage > 0
                          ? 'bg-amber-100 text-amber-800'
                          : 'bg-slate-100 text-slate-600'
                      }`}
                    >
                      {c.missing_percentage}%
                    </span>
                  </td>
                  <td className="px-3 py-2 text-slate-600">{c.unique_count.toLocaleString()}</td>
                  <td className="px-3 py-2 text-slate-600">{c.mean !== null && c.mean !== undefined ? c.mean.toLocaleString() : '-'}</td>
                  <td className="px-3 py-2 text-slate-600">{c.std !== null && c.std !== undefined ? c.std.toLocaleString() : '-'}</td>
                  <td className="px-3 py-2 text-slate-600">{c.min !== null && c.min !== undefined ? c.min.toLocaleString() : '-'}</td>
                  <td className="px-3 py-2 text-slate-600">{c.max !== null && c.max !== undefined ? c.max.toLocaleString() : '-'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
