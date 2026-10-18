import React from 'react';
import { useStudio, WorkflowStage } from '../../context/StudioContext';
import {
  FileText,
  BarChart2,
  Sliders,
  HelpCircle,
  Cpu,
  Layers,
  Search,
  Download,
  ChevronRight,
  CheckCircle2,
} from 'lucide-react';

interface StepItem {
  id: WorkflowStage;
  num: number;
  title: string;
  subtitle: string;
  icon: React.ReactNode;
}

const steps: StepItem[] = [
  { id: 'dataset', num: 1, title: 'Upload Dataset', subtitle: 'CSV / Excel file', icon: <FileText className="w-4 h-4" /> },
  { id: 'profile', num: 2, title: 'Data Profiling', subtitle: 'Dataset info & types', icon: <BarChart2 className="w-4 h-4" /> },
  { id: 'preprocess', num: 3, title: 'Preprocessing', subtitle: 'Clean, encode, scale', icon: <Sliders className="w-4 h-4" /> },
  { id: 'problem', num: 4, title: 'Problem Detection', subtitle: 'Classification/Reg.', icon: <HelpCircle className="w-4 h-4" /> },
  { id: 'train', num: 5, title: 'Train Models', subtitle: 'Multiple ML algorithms', icon: <Cpu className="w-4 h-4" /> },
  { id: 'compare', num: 6, title: 'Evaluate & Compare', subtitle: 'Metrics & ranking', icon: <Layers className="w-4 h-4" /> },
  { id: 'explain', num: 7, title: 'Explain Predictions', subtitle: 'SHAP & feature imp.', icon: <Search className="w-4 h-4" /> },
  { id: 'predict', num: 8, title: 'Predict & Export', subtitle: 'Inference & PDF', icon: <Download className="w-4 h-4" /> },
];

export const Header: React.FC = () => {
  const { activeStage, setActiveStage, completedStages } = useStudio();

  return (
    <header className="bg-slate-900 border-b border-slate-800 text-white select-none">
      {/* Top Banner Title */}
      <div className="px-6 py-2.5 bg-slate-950 flex items-center justify-between border-b border-slate-800/60">
        <div className="flex items-center gap-3">
          <div className="bg-blue-600 p-1.5 rounded-lg text-white font-bold text-xs tracking-wider flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
            STUDIO WORKFLOW
          </div>
          <h1 className="text-xs uppercase tracking-widest font-semibold text-slate-300">
            End-to-End ML Experimentation & Model Evaluation Platform
          </h1>
        </div>
        <div className="text-xs text-slate-400 flex items-center gap-2">
          <span className="px-2 py-0.5 bg-slate-800 rounded text-slate-300 border border-slate-700">Python 3.14 + Scikit-Learn + SHAP</span>
        </div>
      </div>

      {/* Stepper Steps (Matching Reference Image Step Bar) */}
      <div className="px-4 py-3 overflow-x-auto">
        <div className="flex items-center min-w-max gap-2">
          {steps.map((step, idx) => {
            const isActive = activeStage === step.id || (step.id === 'evaluate' && activeStage === 'evaluate');
            const isCompleted = completedStages.has(step.id);
            const isClickable = isCompleted || isActive;

            return (
              <React.Fragment key={step.id}>
                <button
                  type="button"
                  disabled={!isClickable}
                  onClick={() => setActiveStage(step.id)}
                  className={`flex items-center gap-2.5 px-3 py-2 rounded-xl text-left transition-all duration-200 ${
                    isActive
                      ? 'bg-blue-600/20 border-2 border-blue-500 text-white shadow-lg shadow-blue-500/10'
                      : isCompleted
                      ? 'bg-slate-800/80 hover:bg-slate-800 border border-slate-700 text-slate-200 cursor-pointer'
                      : 'bg-slate-900/50 border border-slate-800/80 text-slate-500 opacity-60 cursor-not-allowed'
                  }`}
                >
                  <div
                    className={`w-6 h-6 rounded-full flex items-center justify-center font-bold text-xs shrink-0 ${
                      isActive
                        ? 'bg-blue-600 text-white'
                        : isCompleted
                        ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
                        : 'bg-slate-800 text-slate-500'
                    }`}
                  >
                    {isCompleted && !isActive ? <CheckCircle2 className="w-3.5 h-3.5" /> : step.num}
                  </div>
                  <div>
                    <div className="text-xs font-semibold leading-tight flex items-center gap-1.5">
                      {step.title}
                    </div>
                    <div className="text-[10px] text-slate-400 leading-tight">
                      {step.subtitle}
                    </div>
                  </div>
                </button>

                {idx < steps.length - 1 && (
                  <ChevronRight className="w-3.5 h-3.5 text-slate-600 shrink-0" />
                )}
              </React.Fragment>
            );
          })}
        </div>
      </div>
    </header>
  );
};
