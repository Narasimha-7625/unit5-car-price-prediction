import React from 'react';
import {
  ActualVsPredictedChart,
  CartTreeDiagram,
  FeatureImportanceChart,
  HistogramChart,
} from '../components/Charts.tsx';
import { formatInr, formatLakh } from '../ml/preprocessing.ts';
import type { CartModelArtifact } from '../ml/types.ts';

export function ModelAnalytics({ model }: { model: CartModelArtifact }) {
  const { metrics, bestParams, gridSearchSummary, treeStats } = model;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-slate-900">
          CART Model Analytics & Evaluation
        </h1>
        <p className="text-sm text-slate-600 mt-1">
          All metrics below are computed directly on the untouched {model.trainTestRatio} test split ({metrics.testCount} vehicles, random_state={model.randomState}) after {gridSearchSummary.cvFolds}-fold cross-validation on the {metrics.trainCount} training records.
        </p>
      </div>

      {/* 4 Required Regression Metrics with Plain-Language Explanations */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="bg-white border border-slate-200 rounded-xl p-5 space-y-2">
          <div className="text-xs font-medium text-slate-500">Mean Absolute Error (MAE)</div>
          <div className="text-2xl font-bold font-mono tabular-nums text-slate-900">
            {formatInr(metrics.mae)}
          </div>
          <div className="text-xs font-mono text-blue-800 tabular-nums">
            ({formatLakh(metrics.mae)})
          </div>
          <p className="text-xs text-slate-600 leading-relaxed">
            Average absolute rupee gap between the actual selling price and the CART prediction across all test cars.
          </p>
        </div>

        <div className="bg-white border border-slate-200 rounded-xl p-5 space-y-2">
          <div className="text-xs font-medium text-slate-500">Mean Squared Error (MSE)</div>
          <div className="text-xl font-bold font-mono tabular-nums text-slate-900">
            {Math.round(metrics.mse).toLocaleString('en-IN')}
          </div>
          <div className="text-xs font-mono text-slate-500">INR²</div>
          <p className="text-xs text-slate-600 leading-relaxed">
            Average of squared prediction errors. Penalizes large valuation misses heavily and serves as the CART splitting objective.
          </p>
        </div>

        <div className="bg-white border border-slate-200 rounded-xl p-5 space-y-2">
          <div className="text-xs font-medium text-slate-500">Root Mean Squared Error (RMSE)</div>
          <div className="text-2xl font-bold font-mono tabular-nums text-slate-900">
            {formatInr(metrics.rmse)}
          </div>
          <div className="text-xs font-mono text-blue-800 tabular-nums">
            ({formatLakh(metrics.rmse)})
          </div>
          <p className="text-xs text-slate-600 leading-relaxed">
            Square root of MSE, expressed back in Indian Rupees. More sensitive to occasional luxury-car outliers than MAE.
          </p>
        </div>

        <div className="bg-white border border-slate-200 rounded-xl p-5 space-y-2">
          <div className="text-xs font-medium text-slate-500">R-Squared Score (R²)</div>
          <div className="text-2xl font-bold font-mono tabular-nums text-emerald-700">
            {metrics.r2.toFixed(4)}
          </div>
          <div className="text-xs font-mono text-slate-600 tabular-nums">
            Train R²: {metrics.trainR2.toFixed(4)}
          </div>
          <p className="text-xs text-slate-600 leading-relaxed">
            Proportion of used-car price variance explained by the CART tree on unseen test data (1.00 = perfect fit).
          </p>
        </div>
      </div>

      {/* Important Caveat on R-Squared */}
      <div className="bg-amber-50 border border-amber-200 rounded-xl p-4 text-xs text-amber-950 leading-relaxed">
        <strong>Interpreting R² Responsibly:</strong> Even a high R² score measures overall variance explained across the test set and <strong>does not guarantee accurate predictions for every individual car</strong>. Rare variants, accident history, aftermarket accessories, or localized city demand outside the dataset can still cause individual residuals larger than the average MAE.
      </div>

      {/* Hyperparameters, Split Sizes & GridSearchCV Candidates */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        <div className="lg:col-span-5 bg-white border border-slate-200 rounded-xl p-5 space-y-4">
          <div>
            <h2 className="text-base font-semibold text-slate-900">
              Selected CART Configuration & Split Info
            </h2>
            <p className="text-xs text-slate-500 mt-0.5">
              Optimal parameters chosen by {gridSearchSummary.cvFolds}-fold cross-validation on training data only
            </p>
          </div>

          <dl className="divide-y divide-slate-100 text-xs">
            <div className="py-2 flex justify-between">
              <dt className="text-slate-600">Algorithm</dt>
              <dd className="font-mono font-semibold text-slate-900">DecisionTreeRegressor (CART)</dd>
            </div>
            <div className="py-2 flex justify-between">
              <dt className="text-slate-600">Splitting Criterion</dt>
              <dd className="font-mono font-semibold text-slate-900">squared_error (MSE)</dd>
            </div>
            <div className="py-2 flex justify-between">
              <dt className="text-slate-600">Train / Test Split</dt>
              <dd className="font-mono tabular-nums text-slate-900">
                {metrics.trainCount} train / {metrics.testCount} test ({model.trainTestRatio})
              </dd>
            </div>
            <div className="py-2 flex justify-between">
              <dt className="text-slate-600">Best max_depth</dt>
              <dd className="font-mono tabular-nums font-semibold text-blue-800">{bestParams.max_depth}</dd>
            </div>
            <div className="py-2 flex justify-between">
              <dt className="text-slate-600">Best min_samples_split</dt>
              <dd className="font-mono tabular-nums font-semibold text-blue-800">{bestParams.min_samples_split}</dd>
            </div>
            <div className="py-2 flex justify-between">
              <dt className="text-slate-600">Best min_samples_leaf</dt>
              <dd className="font-mono tabular-nums font-semibold text-blue-800">{bestParams.min_samples_leaf}</dd>
            </div>
            <div className="py-2 flex justify-between">
              <dt className="text-slate-600">Best ccp_alpha (Pruning)</dt>
              <dd className="font-mono tabular-nums font-semibold text-blue-800">
                {bestParams.ccp_alpha.toLocaleString('en-IN')}
              </dd>
            </div>
            <div className="py-2 flex justify-between">
              <dt className="text-slate-600">Final Tree Topology</dt>
              <dd className="font-mono tabular-nums text-slate-900">
                {treeStats.totalNodes} nodes · {treeStats.leafNodes} leaves · depth {treeStats.actualMaxDepth}
              </dd>
            </div>
          </dl>
        </div>

        <div className="lg:col-span-7 bg-white border border-slate-200 rounded-xl p-5 space-y-3">
          <div>
            <h2 className="text-base font-semibold text-slate-900">
              Top GridSearchCV Candidates ({gridSearchSummary.totalCombinationsTested} Tested)
            </h2>
            <p className="text-xs text-slate-500 mt-0.5">
              Ranked by mean {gridSearchSummary.cvFolds}-fold cross-validated R² strictly within the training split
            </p>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead>
                <tr className="border-b border-slate-200 text-slate-500">
                  <th className="py-2 pr-3 font-medium">Rank</th>
                  <th className="py-2 pr-3 font-medium">max_depth</th>
                  <th className="py-2 pr-3 font-medium">min_split</th>
                  <th className="py-2 pr-3 font-medium">min_leaf</th>
                  <th className="py-2 pr-3 font-medium">ccp_alpha</th>
                  <th className="py-2 pr-3 font-medium text-right">Mean CV R²</th>
                  <th className="py-2 font-medium text-right">Mean CV RMSE</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 font-mono tabular-nums">
                {gridSearchSummary.topCandidates.map((c, idx) => (
                  <tr key={idx} className={idx === 0 ? 'bg-blue-50/60 font-semibold' : ''}>
                    <td className="py-2 pr-3 text-slate-700">#{idx + 1}</td>
                    <td className="py-2 pr-3 text-slate-900">{c.params.max_depth}</td>
                    <td className="py-2 pr-3 text-slate-900">{c.params.min_samples_split}</td>
                    <td className="py-2 pr-3 text-slate-900">{c.params.min_samples_leaf}</td>
                    <td className="py-2 pr-3 text-slate-700">
                      {c.params.ccp_alpha.toLocaleString('en-IN')}
                    </td>
                    <td className="py-2 pr-3 text-right text-emerald-700">
                      {c.meanCvR2.toFixed(4)}
                    </td>
                    <td className="py-2 text-right text-slate-900">
                      {formatLakh(c.meanCvRmse)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {/* Actual vs Predicted & Residual Distribution */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <ActualVsPredictedChart points={model.actualVsPredicted} />
        <HistogramChart
          title="Prediction Error (Residual) Distribution"
          subtitle="Residual = Actual Price − Predicted Price across test records (centered near 0 is ideal)"
          bins={model.charts.residualHistogram}
          barColor="#0F172A"
          xAxisLabel="Residual Error (INR in Lakhs)"
        />
      </div>

      {/* Feature Importance */}
      <FeatureImportanceChart items={model.featureImportances} />

      {/* Interactive CART Tree Visualizer */}
      <CartTreeDiagram tree={model.tree} maxDisplayDepth={3} />
    </div>
  );
}
