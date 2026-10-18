import React, { useState, useRef, useEffect } from 'react';
import { useStudio } from '../../context/StudioContext';
import { apiClient } from '../../api/client';
import { UploadCloud, FileSpreadsheet, X, ArrowRight, Loader2, Sparkles } from 'lucide-react';

export const DatasetUploadScreen: React.FC = () => {
  const {
    dataset,
    setDataset,
    targetColumn,
    setTargetColumn,
    idColumn,
    setIdColumn,
    loadDatasetProfile,
    runProblemDetection,
    setActiveStage,
    markStageCompleted,
  } = useStudio();

  const [isUploading, setIsUploading] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [samples, setSamples] = useState<{ filename: string; size_bytes: number; description: string }[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    apiClient.listSampleDatasets().then(setSamples).catch(console.error);
  }, []);

  const handleFileUpload = async (file: File) => {
    setIsUploading(true);
    setErrorMsg(null);
    try {
      const res = await apiClient.uploadDataset(file);
      setDataset(res);
      if (res.suggested_target_column) setTargetColumn(res.suggested_target_column);
      if (res.suggested_id_column) setIdColumn(res.suggested_id_column);
    } catch (err: any) {
      setErrorMsg(err.response?.data?.detail || 'Failed to upload dataset.');
    } finally {
      setIsUploading(false);
    }
  };

  const handleLoadSample = async (filename: string) => {
    setIsUploading(true);
    setErrorMsg(null);
    try {
      const res = await apiClient.loadSampleDataset(filename);
      setDataset(res);
      if (res.suggested_target_column) setTargetColumn(res.suggested_target_column);
      if (res.suggested_id_column) setIdColumn(res.suggested_id_column);
    } catch (err: any) {
      setErrorMsg(err.response?.data?.detail || 'Failed to load sample dataset.');
    } finally {
      setIsUploading(false);
    }
  };

  const handleAnalyze = async () => {
    if (!dataset || !targetColumn) return;
    setIsUploading(true);
    try {
      await loadDatasetProfile(targetColumn, idColumn === 'None' ? undefined : idColumn);
      await runProblemDetection(targetColumn);
      markStageCompleted('dataset');
      markStageCompleted('profile');
      setActiveStage('profile');
    } catch (err: any) {
      setErrorMsg(err.response?.data?.detail || 'Failed to analyze dataset.');
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className="max-w-4xl mx-auto py-8 px-6">
      <div className="text-center mb-8">
        <h2 className="text-2xl font-bold text-slate-900 tracking-tight">Upload Your Dataset</h2>
        <p className="text-sm text-slate-500 mt-1">Upload a CSV or Excel file to get started</p>
      </div>

      {errorMsg && (
        <div className="mb-6 p-4 bg-red-50 border border-red-200 text-red-700 text-sm rounded-xl">
          {errorMsg}
        </div>
      )}

      {/* Main Upload Card */}
      <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-8 space-y-6">
        {/* Drag and Drop Box */}
        <div
          onDragOver={(e) => e.preventDefault()}
          onDrop={(e) => {
            e.preventDefault();
            if (e.dataTransfer.files && e.dataTransfer.files[0]) {
              handleFileUpload(e.dataTransfer.files[0]);
            }
          }}
          className="border-2 border-dashed border-slate-300 hover:border-blue-500 rounded-2xl p-10 flex flex-col items-center justify-center text-center bg-slate-50/50 hover:bg-blue-50/20 transition-all cursor-pointer"
          onClick={() => fileInputRef.current?.click()}
        >
          <input
            type="file"
            ref={fileInputRef}
            onChange={(e) => {
              if (e.target.files && e.target.files[0]) {
                handleFileUpload(e.target.files[0]);
              }
            }}
            accept=".csv,.xlsx,.xls"
            className="hidden"
          />

          <div className="w-16 h-16 rounded-full bg-blue-100 flex items-center justify-center text-blue-600 mb-4 shadow-sm">
            <UploadCloud className="w-8 h-8" />
          </div>

          <h3 className="font-semibold text-slate-800 text-base mb-1">
            Drag and drop your file here
          </h3>
          <p className="text-slate-500 text-xs mb-4">or</p>

          <button
            type="button"
            className="px-5 py-2 bg-blue-600 hover:bg-blue-700 text-white text-xs font-semibold rounded-lg shadow-sm transition"
          >
            Browse Files
          </button>

          <p className="text-[11px] text-slate-400 mt-4">Supported formats: .csv, .xlsx</p>
        </div>

        {/* Quick Sample Dataset Selector */}
        {samples.length > 0 && (
          <div className="pt-2 border-t border-slate-100">
            <div className="flex items-center gap-1.5 text-xs font-semibold text-slate-600 mb-3">
              <Sparkles className="w-3.5 h-3.5 text-amber-500" />
              <span>Or try pre-loaded production sample datasets:</span>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              {samples.map((s) => (
                <button
                  key={s.filename}
                  type="button"
                  onClick={() => handleLoadSample(s.filename)}
                  className="flex items-center justify-between p-3 rounded-xl border border-slate-200 hover:border-blue-400 hover:bg-blue-50/30 transition text-left"
                >
                  <div className="flex items-center gap-3">
                    <FileSpreadsheet className="w-5 h-5 text-emerald-600 shrink-0" />
                    <div>
                      <div className="font-medium text-xs text-slate-800">{s.filename}</div>
                      <div className="text-[10px] text-slate-500">{s.description}</div>
                    </div>
                  </div>
                  <span className="text-[10px] font-semibold text-blue-600 shrink-0 px-2 py-0.5 bg-blue-50 rounded">
                    Load
                  </span>
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Selected File Card */}
        {dataset && (
          <div className="pt-4 border-t border-slate-100 space-y-6">
            <div className="flex items-center justify-between p-4 bg-slate-50 border border-slate-200 rounded-xl">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-lg bg-emerald-100 text-emerald-700 flex items-center justify-center font-bold text-xs">
                  CSV
                </div>
                <div>
                  <h4 className="font-semibold text-slate-800 text-xs">{dataset.filename}</h4>
                  <span className="text-[11px] text-slate-500">
                    {(dataset.file_size_bytes / (1024 * 1024)).toFixed(1)} MB • {dataset.row_count} rows • {dataset.col_count} columns
                  </span>
                </div>
              </div>
              <button
                type="button"
                onClick={() => setDataset(null)}
                className="text-slate-400 hover:text-slate-600 p-1"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            {/* Target and ID Columns Grid matching screenshot */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block text-xs font-semibold text-slate-700 mb-1.5">
                  Select Target Column <span className="text-red-500">*</span>
                </label>
                <select
                  value={targetColumn}
                  onChange={(e) => setTargetColumn(e.target.value)}
                  className="w-full bg-white border border-slate-300 rounded-xl px-3 py-2 text-xs font-medium text-slate-800 focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  <option value="">-- Choose Target Column --</option>
                  {dataset.columns.map((col) => (
                    <option key={col} value={col}>
                      {col} {col === dataset.suggested_target_column ? '(Auto-suggested)' : ''}
                    </option>
                  ))}
                </select>
                <p className="text-[11px] text-slate-400 mt-1">
                  The column model will learn to predict.
                </p>
              </div>

              <div>
                <label className="block text-xs font-semibold text-slate-700 mb-1.5">
                  Exclude ID Column (optional)
                </label>
                <select
                  value={idColumn}
                  onChange={(e) => setIdColumn(e.target.value)}
                  className="w-full bg-white border border-slate-300 rounded-xl px-3 py-2 text-xs font-medium text-slate-800 focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  <option value="None">None</option>
                  {dataset.columns
                    .filter((c) => c !== targetColumn)
                    .map((col) => (
                      <option key={col} value={col}>
                        {col} {col === dataset.suggested_id_column ? '(Auto-detected)' : ''}
                      </option>
                    ))}
                </select>
                <p className="text-[11px] text-slate-400 mt-1">
                  Unique identifier column to drop before training.
                </p>
              </div>
            </div>

            {/* Action button matching screenshot */}
            <div className="flex justify-end pt-2">
              <button
                type="button"
                disabled={!targetColumn || isUploading}
                onClick={handleAnalyze}
                className="flex items-center gap-2 px-6 py-2.5 bg-blue-600 hover:bg-blue-700 disabled:opacity-50 text-white font-semibold text-xs rounded-xl shadow-md transition"
              >
                {isUploading ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin" />
                    Analyzing Dataset...
                  </>
                ) : (
                  <>
                    Analyze Dataset
                    <ArrowRight className="w-4 h-4" />
                  </>
                )}
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
