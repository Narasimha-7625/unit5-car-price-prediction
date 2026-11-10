import React, { useState } from 'react';
import { ArrowRight, BarChart3, Calculator, Database, GitBranch } from 'lucide-react';
import { formatInr, formatLakh } from '../ml/preprocessing.ts';
import type { CartModelArtifact } from '../ml/types.ts';

export type PageId =
  | 'home'
  | 'predict'
  | 'dataset'
  | 'analytics'
  | 'about'
  | 'project';

export function HomeDashboard({
  model,
  onNavigate,
}: {
  model: CartModelArtifact;
  onNavigate: (page: PageId) => void;
}) {
  const [imgError, setImgError] = useState(false);

  return (
    <div className="space-y-8">
      {/* Editorial 2-Column Hero Section */}
      <section className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-stretch">
        <div className="lg:col-span-7 bg-white border border-slate-200 rounded-xl p-8 flex flex-col justify-between">
          <div className="space-y-4">
            <div className="flex flex-wrap items-center gap-2 text-xs text-slate-500">
              <span>Academic Machine Learning Project</span>
              <span aria-hidden="true">·</span>
              <span>Indian Used-Car Market</span>
              <span aria-hidden="true">·</span>
              <span>Unit 5 CART Regression</span>
            </div>
            <h1 className="text-3xl lg:text-4xl font-bold tracking-tight text-slate-900 leading-tight [text-wrap:balance]">
              Used Car Price Prediction System Using CART Regression
            </h1>
            <p className="text-base text-slate-600 leading-relaxed max-w-2xl">
              An interactive valuation and model analysis platform powered exclusively by Scikit-learn style Classification and Regression Trees (DecisionTreeRegressor). Inspect real data preprocessing, cross-validated hyperparameter tuning, and transparent decision rules behind every Indian Rupee estimate.
            </p>
          </div>

          <div className="mt-8 flex flex-wrap items-center gap-3">
            <button
              onClick={() => onNavigate('predict')}
              className="inline-flex items-center gap-2 px-5 py-2.5 text-sm font-semibold text-white bg-blue-800 hover:bg-blue-900 rounded-lg transition-colors whitespace-nowrap shrink-0"
            >
              <Calculator className="w-4 h-4" />
              Predict Car Price
              <ArrowRight className="w-4 h-4" />
            </button>
            <button
              onClick={() => onNavigate('analytics')}
              className="inline-flex items-center gap-2 px-5 py-2.5 text-sm font-medium text-slate-700 bg-white border border-slate-300 hover:bg-slate-50 rounded-lg transition-colors whitespace-nowrap shrink-0"
            >
              <BarChart3 className="w-4 h-4" />
              Explore Model Analytics
            </button>
          </div>
        </div>

        <div className="lg:col-span-5 bg-white border border-slate-200 rounded-xl overflow-hidden min-h-[260px] flex">
          {!imgError ? (
            <img
              src="/src/assets/images/hero_indian_car_studio_1794325204033.jpg"
              alt="Modern silver compact SUV inside a minimalist architectural automotive studio"
              referrerPolicy="no-referrer"
              onError={() => setImgError(true)}
              className="w-full h-full object-cover"
            />
          ) : (
            <div className="w-full h-full bg-slate-100 flex flex-col items-center justify-center p-6 text-center">
              <GitBranch className="w-10 h-10 text-slate-400 mb-2" />
              <span className="text-sm font-medium text-slate-700">
                Indian Used-Car Valuation Studio
              </span>
            </div>
          )}
        </div>
      </section>

      {/* Honest Dataset Provenance Banner */}
      <section className="bg-white border border-slate-200 rounded-xl p-5 flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div className="space-y-1">
          <div className="text-xs font-medium text-slate-500">
            Active Training Dataset Source
          </div>
          <div className="text-sm font-semibold text-slate-900">
            {model.datasetName}
          </div>
          <p className="text-xs text-slate-600 max-w-3xl">
            {model.datasetSource === 'sample_demo'
              ? 'Currently using the built-in Sample Demonstration Dataset for immediate evaluation. Upload a real Indian used-car CSV (e.g., Kaggle CarDekho dataset) in Dataset Explorer to train on full market records.'
              : 'Model trained on your uploaded CSV dataset with zero data leakage and 5-fold cross-validated CART tuning.'}
          </p>
        </div>
        <button
          onClick={() => onNavigate('dataset')}
          className="inline-flex items-center gap-2 px-4 py-2 text-xs font-semibold text-slate-800 bg-slate-100 hover:bg-slate-200 rounded-lg transition-colors whitespace-nowrap shrink-0"
        >
          <Database className="w-3.5 h-3.5" />
          Manage Dataset & Upload CSV
        </button>
      </section>

      {/* 4-Column Actual Metrics Strip */}
      <section className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="bg-white border border-slate-200 rounded-xl p-5">
          <div className="text-xs text-slate-500">Training Records (80% Split)</div>
          <div className="mt-2 text-2xl font-bold text-slate-900 font-mono tabular-nums">
            {model.metrics.trainCount.toLocaleString('en-IN')}
          </div>
          <div className="mt-1 text-xs text-slate-600 tabular-nums">
            {model.metrics.testCount.toLocaleString('en-IN')} untouched test records · {model.totalCleanedRows.toLocaleString('en-IN')} cleaned total
          </div>
        </div>

        <div className="bg-white border border-slate-200 rounded-xl p-5">
          <div className="text-xs text-slate-500">Selected Algorithm</div>
          <div className="mt-2 text-xl font-bold text-slate-900">
            CART Regressor
          </div>
          <div className="mt-1 text-xs text-slate-600 font-mono tabular-nums">
            max_depth={model.bestParams.max_depth} · leaf={model.bestParams.min_samples_leaf} · split={model.bestParams.min_samples_split}
          </div>
        </div>

        <div className="bg-white border border-slate-200 rounded-xl p-5">
          <div className="text-xs text-slate-500">Test R-Squared (R²)</div>
          <div className="mt-2 text-2xl font-bold text-emerald-700 font-mono tabular-nums">
            {(model.metrics.r2 * 100).toFixed(1)}%
          </div>
          <div className="mt-1 text-xs text-slate-600 font-mono tabular-nums">
            Score: {model.metrics.r2.toFixed(4)} (Train: {model.metrics.trainR2.toFixed(4)})
          </div>
        </div>

        <div className="bg-white border border-slate-200 rounded-xl p-5">
          <div className="text-xs text-slate-500">Test Prediction Error (MAE)</div>
          <div className="mt-2 text-2xl font-bold text-slate-900 font-mono tabular-nums">
            {formatLakh(model.metrics.mae)}
          </div>
          <div className="mt-1 text-xs text-slate-600 font-mono tabular-nums">
            MAE {formatInr(model.metrics.mae)} · RMSE {formatLakh(model.metrics.rmse)}
          </div>
        </div>
      </section>

      {/* How the System Works */}
      <section className="bg-white border border-slate-200 rounded-xl p-6 space-y-5">
        <div>
          <h2 className="text-lg font-semibold text-slate-900">
            How the CART Valuation Pipeline Works
          </h2>
          <p className="text-sm text-slate-600 mt-1">
            Every resale estimate follows a leakage-free, reproducible machine learning pipeline built strictly on DecisionTreeRegressor.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 pt-2 border-t border-slate-100">
          <div className="space-y-1.5">
            <div className="text-xs font-mono font-semibold text-blue-800">01. Data Cleaning & Encoding</div>
            <h3 className="text-sm font-semibold text-slate-900">Indian Unit & Outlier Normalization</h3>
            <p className="text-xs text-slate-600 leading-relaxed">
              Parses strings like "21.2 kmpl", "1197 CC", and "₹5.45 Lakh", removes exact duplicates and impossible specs, and fits median/one-hot transformers strictly on the 80% training split.
            </p>
          </div>
          <div className="space-y-1.5">
            <div className="text-xs font-mono font-semibold text-blue-800">02. Cross-Validated CART Tuning</div>
            <h3 className="text-sm font-semibold text-slate-900">Squared-Error Splitting & Pruning</h3>
            <p className="text-xs text-slate-600 leading-relaxed">
              Evaluates {model.gridSearchSummary.totalCombinationsTested} hyperparameter combinations via {model.gridSearchSummary.cvFolds}-fold cross-validation across depth, leaf size, and cost-complexity pruning (ccp_alpha).
            </p>
          </div>
          <div className="space-y-1.5">
            <div className="text-xs font-mono font-semibold text-blue-800">03. Explainable Leaf Valuation</div>
            <h3 className="text-sm font-semibold text-slate-900">Traceable Decision Path</h3>
            <p className="text-xs text-slate-600 leading-relaxed">
              Routes your car through binary yes/no thresholds down to a terminal leaf node, returning the historical leaf mean in INR alongside every decision rule traversed.
            </p>
          </div>
        </div>
      </section>
    </div>
  );
}
