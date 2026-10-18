import React from 'react';
import { useStudio, WorkflowStage } from '../../context/StudioContext';
import {
  FileSpreadsheet,
  BarChart3,
  SlidersHorizontal,
  Compass,
  LineChart,
  GitCompare,
  Sparkles,
  PlayCircle,
  DownloadCloud,
  CheckCircle,
} from 'lucide-react';

interface NavItem {
  id: WorkflowStage | 'export_shortcut';
  stage: WorkflowStage;
  label: string;
  icon: React.ReactNode;
}

const navItems: NavItem[] = [
  { id: 'dataset', stage: 'dataset', label: 'Dataset', icon: <FileSpreadsheet className="w-4 h-4" /> },
  { id: 'profile', stage: 'profile', label: 'Data Profile', icon: <BarChart3 className="w-4 h-4" /> },
  { id: 'preprocess', stage: 'preprocess', label: 'Preprocess', icon: <SlidersHorizontal className="w-4 h-4" /> },
  { id: 'problem', stage: 'problem', label: 'Models', icon: <Compass className="w-4 h-4" /> },
  { id: 'evaluate', stage: 'evaluate', label: 'Evaluate', icon: <LineChart className="w-4 h-4" /> },
  { id: 'compare', stage: 'compare', label: 'Compare', icon: <GitCompare className="w-4 h-4" /> },
  { id: 'explain', stage: 'explain', label: 'Explain', icon: <Sparkles className="w-4 h-4" /> },
  { id: 'predict', stage: 'predict', label: 'Predict', icon: <PlayCircle className="w-4 h-4" /> },
  { id: 'export_shortcut', stage: 'predict', label: 'Export', icon: <DownloadCloud className="w-4 h-4" /> },
];

export const Sidebar: React.FC = () => {
  const { activeStage, setActiveStage, completedStages } = useStudio();

  return (
    <aside className="w-64 bg-slate-900 border-r border-slate-800 flex flex-col shrink-0 min-h-screen text-slate-300">
      {/* Brand logo matching the screenshot */}
      <div className="p-5 border-b border-slate-800/80 flex items-center gap-3">
        <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-blue-700 to-indigo-500 flex items-center justify-center text-white shadow-md shadow-blue-500/20 font-bold">
          <span className="text-lg">ML</span>
        </div>
        <div>
          <h2 className="font-bold text-sm tracking-wide text-white leading-tight">
            ML Model
          </h2>
          <span className="text-[11px] text-blue-400 font-medium">Evaluation Studio</span>
        </div>
      </div>

      {/* Nav list */}
      <nav className="p-3 space-y-1 flex-1">
        {navItems.map((item) => {
          const isActive =
            activeStage === item.stage ||
            (item.id === 'problem' && (activeStage === 'problem' || activeStage === 'train')) ||
            (item.id === 'export_shortcut' && activeStage === 'predict');
          const isCompleted = completedStages.has(item.stage);

          return (
            <button
              key={item.id}
              onClick={() => setActiveStage(item.stage)}
              className={`w-full flex items-center justify-between px-3.5 py-2.5 rounded-xl font-medium text-xs transition-all duration-150 ${
                isActive
                  ? 'bg-blue-600 text-white shadow-sm font-semibold'
                  : isCompleted
                  ? 'text-slate-300 hover:bg-slate-800 hover:text-white'
                  : 'text-slate-500 hover:bg-slate-800/40 hover:text-slate-400'
              }`}
            >
              <div className="flex items-center gap-3">
                <span className={isActive ? 'text-white' : isCompleted ? 'text-blue-400' : 'text-slate-500'}>
                  {item.icon}
                </span>
                <span>{item.label}</span>
              </div>
              {isCompleted && !isActive && (
                <CheckCircle className="w-3.5 h-3.5 text-emerald-400" />
              )}
            </button>
          );
        })}
      </nav>

      {/* Footer Info */}
      <div className="p-4 border-t border-slate-800/80 text-[11px] text-slate-500">
        <div className="flex items-center justify-between mb-1">
          <span className="font-semibold text-slate-400">Environment</span>
          <span className="text-emerald-400 font-medium">Ready</span>
        </div>
        <p className="text-[10px] leading-relaxed">
          Real Scikit-learn Pipeline with fitted transformer caching.
        </p>
      </div>
    </aside>
  );
};
