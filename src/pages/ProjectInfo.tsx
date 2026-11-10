import React, { useEffect, useState } from 'react';
import { Check, Copy, FileCode2 } from 'lucide-react';
import type { CartModelArtifact } from '../ml/types.ts';

interface PythonFileItem {
  path: string;
  language: string;
  content: string;
}

export function ProjectInfo({ model }: { model: CartModelArtifact }) {
  const [pyFiles, setPyFiles] = useState<PythonFileItem[]>([]);
  const [activeFilePath, setActiveFilePath] = useState<string>('app.py');
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    fetch('/api/python-files')
      .then((r) => r.json())
      .then((data) => {
        if (Array.isArray(data.files) && data.files.length > 0) {
          setPyFiles(data.files);
          setActiveFilePath(data.files[0].path);
        }
      })
      .catch(() => {
        // Ignore if unavailable
      });
  }, []);

  const activeFile = pyFiles.find((f) => f.path === activeFilePath) ?? pyFiles[0];

  const handleCopy = async () => {
    if (!activeFile) return;
    await navigator.clipboard.writeText(activeFile.content);
    setCopied(true);
    setTimeout(() => setCopied(false), 1800);
  };

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-2xl font-bold text-slate-900">
          Project Information & Academic Documentation
        </h1>
        <p className="text-sm text-slate-600 mt-1">
          Complete engineering report, methodology, limitations, and downloadable Python / Scikit-learn / Streamlit source code modules.
        </p>
      </div>

      {/* Abstract & Problem Statement */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-white border border-slate-200 rounded-xl p-6 space-y-2.5">
          <h2 className="text-base font-semibold text-slate-900">Project Abstract</h2>
          <p className="text-sm text-slate-600 leading-relaxed">
            India's pre-owned automobile market exhibits wide price variance driven by vehicle age, odometer reading, fuel architecture, transmission type, engine displacement, and ownership history. This project implements an end-to-end <strong>Used Car Price Prediction System</strong> strictly utilizing the <strong>Classification and Regression Trees (CART)</strong> algorithm via <code className="font-mono text-xs bg-slate-100 px-1 py-0.5 rounded">DecisionTreeRegressor</code>. The system covers leakage-free preprocessing, cross-validated hyperparameter tuning, quantitative test evaluation, visual tree interpretation, and real-time valuation in Indian Rupees (₹).
          </p>
        </div>

        <div className="bg-white border border-slate-200 rounded-xl p-6 space-y-2.5">
          <h2 className="text-base font-semibold text-slate-900">Problem Statement</h2>
          <p className="text-sm text-slate-600 leading-relaxed">
            Buyers and sellers in the Indian used-car ecosystem frequently rely on subjective dealer quotes or rough depreciation rules of thumb, resulting in inconsistent pricing. A transparent, data-driven regression model is needed that can process mixed numerical and categorical vehicle specifications, explain how each price decision is reached, and adapt to real-world Indian CSV datasets containing currency units and missing entries.
          </p>
        </div>
      </div>

      {/* Objectives & Methodology */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-white border border-slate-200 rounded-xl p-6 space-y-3">
          <h2 className="text-base font-semibold text-slate-900">Project Objectives</h2>
          <ul className="text-sm text-slate-600 space-y-2 list-disc list-inside leading-relaxed">
            <li>Demonstrate the mathematical and practical operation of CART regression without mixing ensemble or neural algorithms.</li>
            <li>Preprocess real Indian used-car CSVs cleanly, parsing unit strings (kmpl, CC, bhp, ₹, Lakh) and filtering duplicates/invalid entries.</li>
            <li>Prevent data leakage by splitting 80:20 before fitting imputers, categorical encoders, and 5-fold GridSearchCV tuning.</li>
            <li>Evaluate generalization using MAE, MSE, RMSE, R², actual-vs-predicted plots, residual histograms, and feature importances.</li>
            <li>Provide an interactive valuation interface with strict input validation and explainable root-to-leaf decision paths.</li>
          </ul>
        </div>

        <div className="bg-white border border-slate-200 rounded-xl p-6 space-y-3">
          <h2 className="text-base font-semibold text-slate-900">Methodology & Algorithm Summary</h2>
          <ol className="text-sm text-slate-600 space-y-2 list-decimal list-inside leading-relaxed">
            <li><strong>Data Ingestion & Cleaning:</strong> Parse CSV, normalize currency & text metrics, remove exact duplicates and impossible specs.</li>
            <li><strong>Reproducible Split:</strong> Partition cleaned data 80% train / 20% test (<code className="font-mono text-xs">random_state=42</code>).</li>
            <li><strong>Leak-Free ColumnTransformer:</strong> Median imputation for numeric columns; unknown-safe one-hot encoding for categoricals fit only on train.</li>
            <li><strong>CART Tuning (GridSearchCV):</strong> Tune <code className="font-mono text-xs">max_depth</code>, <code className="font-mono text-xs">min_samples_split</code>, <code className="font-mono text-xs">min_samples_leaf</code>, and <code className="font-mono text-xs">ccp_alpha</code> via 5-fold CV.</li>
            <li><strong>Persistence & Inference:</strong> Persist trained pipeline (<code className="font-mono text-xs">cart_model.joblib</code> / <code className="font-mono text-xs">cart_model.json</code>) and serve live predictions.</li>
          </ol>
        </div>
      </div>

      {/* Technology Stack, Limitations, Future Enhancements */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-white border border-slate-200 rounded-xl p-5 space-y-2.5">
          <h2 className="text-base font-semibold text-slate-900">Technology Stack</h2>
          <ul className="text-xs text-slate-600 space-y-1.5 leading-relaxed">
            <li><strong>Python Academic Stack:</strong> Python 3.10+, Scikit-learn (<code className="font-mono">DecisionTreeRegressor</code>, <code className="font-mono">GridSearchCV</code>, <code className="font-mono">ColumnTransformer</code>), Pandas, NumPy, Joblib, Plotly, Matplotlib, Streamlit.</li>
            <li><strong>Live Web Runtime:</strong> React 19, TypeScript, Tailwind CSS, Express.js, custom CART engine executing identically in-browser/server on port 3000.</li>
            <li><strong>Current Trained Features:</strong> {model.preprocessorState.activeNumericFeatures.length} numeric, {model.preprocessorState.activeCategoricalFeatures.length} categorical.</li>
          </ul>
        </div>

        <div className="bg-white border border-slate-200 rounded-xl p-5 space-y-2.5">
          <h2 className="text-base font-semibold text-slate-900">Limitations</h2>
          <ul className="text-xs text-slate-600 space-y-1.5 list-disc list-inside leading-relaxed">
            <li>Single regression trees exhibit step-function predictions (piecewise constant leaves) rather than smooth interpolation.</li>
            <li>Tree structure can be sensitive to small perturbations in training samples compared to ensemble methods.</li>
            <li>Physical vehicle condition, accident history, service records, and city-level RTO taxes are absent unless provided in the CSV.</li>
            <li>High-cardinality model names outside top training categories fall back to general brand/year/engine splits.</li>
          </ul>
        </div>

        <div className="bg-white border border-slate-200 rounded-xl p-5 space-y-2.5">
          <h2 className="text-base font-semibold text-slate-900">Future Enhancements</h2>
          <ul className="text-xs text-slate-600 space-y-1.5 list-disc list-inside leading-relaxed">
            <li>Incorporate city/RTO state registration and insurance validity columns for regional Indian price calibration.</li>
            <li>Add prediction confidence intervals using leaf sample quantiles (25th–75th percentile ranges).</li>
            <li>Support batch CSV valuation exports for dealership inventory pricing.</li>
            <li>Compare pruned CART trees against post-Unit-5 ensemble methods in future academic extensions.</li>
          </ul>
        </div>
      </div>

      {/* Python / Scikit-learn / Streamlit Source Code Deliverable Viewer */}
      {pyFiles.length > 0 && (
        <div className="bg-white border border-slate-200 rounded-xl p-6 space-y-4">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div>
              <h2 className="text-base font-semibold text-slate-900 flex items-center gap-2">
                <FileCode2 className="w-4 h-4 text-blue-800" />
                Python, Scikit-learn & Streamlit Project Files
              </h2>
              <p className="text-xs text-slate-500 mt-0.5">
                Complete modular Python codebase located in <code className="font-mono">python_streamlit/</code> for local execution with Streamlit and Scikit-learn
              </p>
            </div>
            <button
              onClick={handleCopy}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-slate-700 bg-slate-100 hover:bg-slate-200 rounded-lg transition-colors whitespace-nowrap shrink-0"
            >
              {copied ? <Check className="w-3.5 h-3.5 text-emerald-700" /> : <Copy className="w-3.5 h-3.5" />}
              {copied ? 'Copied' : 'Copy File'}
            </button>
          </div>

          <div className="flex flex-wrap gap-1.5 border-b border-slate-200 pb-3">
            {pyFiles.map((f) => (
              <button
                key={f.path}
                onClick={() => setActiveFilePath(f.path)}
                className={`px-3 py-1.5 text-xs font-mono rounded-md transition-colors whitespace-nowrap shrink-0 ${
                  activeFilePath === f.path
                    ? 'bg-blue-800 text-white'
                    : 'bg-slate-100 text-slate-700 hover:bg-slate-200'
                }`}
              >
                {f.path}
              </button>
            ))}
          </div>

          {activeFile && (
            <pre className="bg-slate-900 text-slate-100 rounded-lg p-4 text-xs font-mono overflow-x-auto max-h-[460px] leading-relaxed">
              <code>{activeFile.content}</code>
            </pre>
          )}
        </div>
      )}
    </div>
  );
}
