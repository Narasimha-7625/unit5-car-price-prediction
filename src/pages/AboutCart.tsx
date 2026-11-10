import React, { useMemo, useState } from 'react';
import { CartTreeDiagram } from '../components/Charts.tsx';
import { formatInr, formatLakh } from '../ml/preprocessing.ts';
import type { CartModelArtifact } from '../ml/types.ts';

interface MiniCar {
  car: string;
  year: number;
  km: number;
  priceLakh: number;
}

const MINI_DATASET: MiniCar[] = [
  { car: 'Alto 800', year: 2016, km: 58000, priceLakh: 2.35 },
  { car: 'Wagon R', year: 2018, km: 44000, priceLakh: 3.95 },
  { car: 'Swift VXi', year: 2019, km: 34000, priceLakh: 5.45 },
  { car: 'Baleno Delta', year: 2020, km: 29000, priceLakh: 6.25 },
  { car: 'Dzire ZXi+', year: 2022, km: 15500, priceLakh: 8.10 },
  { car: 'Brezza VXi', year: 2023, km: 11000, priceLakh: 9.85 },
];

const CANDIDATE_SPLITS = [
  { id: 'yr_2018', label: 'Year <= 2018.5', test: (c: MiniCar) => c.year <= 2018.5 },
  { id: 'yr_2019', label: 'Year <= 2019.5', test: (c: MiniCar) => c.year <= 2019.5 },
  { id: 'yr_2021', label: 'Year <= 2021.0', test: (c: MiniCar) => c.year <= 2021.0 },
  { id: 'km_31k', label: 'Km <= 31,500', test: (c: MiniCar) => c.km <= 31500 },
];

function calcSse(values: number[]): { mean: number; sse: number; mse: number } {
  if (values.length === 0) return { mean: 0, sse: 0, mse: 0 };
  const mean = values.reduce((a, b) => a + b, 0) / values.length;
  const sse = values.reduce((acc, v) => acc + (v - mean) * (v - mean), 0);
  return { mean, sse, mse: sse / values.length };
}

export function AboutCart({ model }: { model: CartModelArtifact }) {
  const [selectedSplitId, setSelectedSplitId] = useState('yr_2019');

  const splitMath = useMemo(() => {
    const parentStats = calcSse(MINI_DATASET.map((c) => c.priceLakh));
    const candidate = CANDIDATE_SPLITS.find((s) => s.id === selectedSplitId) ?? CANDIDATE_SPLITS[1];
    const left = MINI_DATASET.filter(candidate.test);
    const right = MINI_DATASET.filter((c) => !candidate.test(c));
    const leftStats = calcSse(left.map((c) => c.priceLakh));
    const rightStats = calcSse(right.map((c) => c.priceLakh));
    const weightedChildSse = leftStats.sse + rightStats.sse;
    const sseReduction = parentStats.sse - weightedChildSse;

    const allRanked = CANDIDATE_SPLITS.map((s) => {
      const l = calcSse(MINI_DATASET.filter(s.test).map((c) => c.priceLakh));
      const r = calcSse(MINI_DATASET.filter((c) => !s.test(c)).map((c) => c.priceLakh));
      return { id: s.id, label: s.label, reduction: parentStats.sse - (l.sse + r.sse) };
    }).sort((a, b) => b.reduction - a.reduction);

    return {
      candidate,
      parentStats,
      left,
      right,
      leftStats,
      rightStats,
      weightedChildSse,
      sseReduction,
      bestId: allRanked[0].id,
      allRanked,
    };
  }, [selectedSplitId]);

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-2xl font-bold text-slate-900">
          Understanding the CART Regression Algorithm
        </h1>
        <p className="text-sm text-slate-600 mt-1 max-w-3xl">
          Complete Unit 5 academic guide to Classification and Regression Trees (Breiman et al., 1984), mathematical split selection, Gini vs Squared Error, and generalization control.
        </p>
      </div>

      {/* Concepts 1-4 */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-white border border-slate-200 rounded-xl p-6 space-y-2.5">
          <div className="text-xs font-mono font-semibold text-blue-800">01. Definition</div>
          <h2 className="text-base font-semibold text-slate-900">What is CART?</h2>
          <p className="text-sm text-slate-600 leading-relaxed">
            <strong>CART (Classification and Regression Trees)</strong> is a supervised non-parametric learning algorithm introduced by Leo Breiman, Jerome Friedman, Richard Olshen, and Charles Stone (1984). It constructs strictly <strong>binary decision trees</strong> (every internal node splits into exactly two child nodes) to predict either discrete classes (Classification Tree) or continuous numeric quantities (Regression Tree).
          </p>
        </div>

        <div className="bg-white border border-slate-200 rounded-xl p-6 space-y-2.5">
          <div className="text-xs font-mono font-semibold text-blue-800">02. Domain Fit</div>
          <h2 className="text-base font-semibold text-slate-900">
            Why CART Fits Used Car Price Prediction
          </h2>
          <p className="text-sm text-slate-600 leading-relaxed">
            Used-car pricing is inherently <strong>non-linear and conditional</strong>: a 2019 Diesel SUV depreciates differently than a 2019 Petrol hatchback, and brand/transmission interactions create price tiers. CART naturally models these feature interactions without assuming linear equations or scaling features, while producing human-readable rules.
          </p>
        </div>

        <div className="bg-white border border-slate-200 rounded-xl p-6 space-y-2.5">
          <div className="text-xs font-mono font-semibold text-blue-800">03. Predictions</div>
          <h2 className="text-base font-semibold text-slate-900">
            How a Regression Tree Makes Predictions
          </h2>
          <p className="text-sm text-slate-600 leading-relaxed">
            During training, CART recursively partitions the feature space into disjoint rectangular regions <span className="font-mono">R₁, R₂, …, Rₘ</span> (terminal leaves). At inference time, a new car starts at the root node and tests one feature threshold at each split (<span className="font-mono">xⱼ ≤ s</span> goes left, <span className="font-mono">xⱼ &gt; s</span> goes right) until reaching a leaf <span className="font-mono">Rₘ</span>. Its predicted price is the <strong>arithmetic mean</strong> of training target prices inside that leaf: <span className="font-mono">ŷ = (1 / |Rₘ|) ∑ yᵢ</span>.
          </p>
        </div>

        <div className="bg-white border border-slate-200 rounded-xl p-6 space-y-2.5">
          <div className="text-xs font-mono font-semibold text-blue-800">04. Splitting Math</div>
          <h2 className="text-base font-semibold text-slate-900">
            Squared-Error Reduction Split Criterion
          </h2>
          <p className="text-sm text-slate-600 leading-relaxed">
            To pick the best feature <span className="font-mono">j</span> and threshold <span className="font-mono">s</span> at node <span className="font-mono">t</span>, CART minimizes the sum of squared errors (SSE) in the resulting children:
          </p>
          <div className="p-3 bg-slate-50 border border-slate-200 rounded-lg font-mono text-xs text-slate-800">
            SSE(t) = ∑ᵢ∈ₜ (yᵢ − ȳₜ)² &nbsp;&nbsp;|&nbsp;&nbsp; ΔSSE = SSE(t) − [SSE(t_L) + SSE(t_R)]
          </div>
          <p className="text-xs text-slate-600">
            Equivalently, Scikit-learn maximizes weighted MSE variance reduction across all candidate split points.
          </p>
        </div>
      </div>

      {/* Classification vs Regression Comparison Table + Gini Explanation */}
      <div className="bg-white border border-slate-200 rounded-xl p-6 space-y-5">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          <div className="lg:col-span-7 space-y-3">
            <h2 className="text-base font-semibold text-slate-900">
              05. Classification Trees vs Regression Trees
            </h2>
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead>
                  <tr className="border-b border-slate-200 text-slate-500">
                    <th className="py-2 pr-4 font-medium">Dimension</th>
                    <th className="py-2 pr-4 font-medium">Classification Tree</th>
                    <th className="py-2 font-medium">Regression Tree (Used Here)</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100 text-slate-700">
                  <tr>
                    <td className="py-2 pr-4 font-medium text-slate-900">Target Variable</td>
                    <td className="py-2 pr-4">Discrete class label (e.g., Buy / Skip)</td>
                    <td className="py-2">Continuous numeric value (Price in ₹)</td>
                  </tr>
                  <tr>
                    <td className="py-2 pr-4 font-medium text-slate-900">Splitting Criterion</td>
                    <td className="py-2 pr-4">Gini Impurity or Entropy</td>
                    <td className="py-2">Squared Error / Variance Reduction (MSE)</td>
                  </tr>
                  <tr>
                    <td className="py-2 pr-4 font-medium text-slate-900">Leaf Output</td>
                    <td className="py-2 pr-4">Majority class vote</td>
                    <td className="py-2">Mean target value of leaf samples</td>
                  </tr>
                  <tr>
                    <td className="py-2 pr-4 font-medium text-slate-900">Evaluation Metrics</td>
                    <td className="py-2 pr-4">Accuracy, Precision, Recall, F1</td>
                    <td className="py-2">MAE, MSE, RMSE, R²</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>

          <div className="lg:col-span-5 space-y-2.5 bg-slate-50 border border-slate-200 rounded-lg p-4">
            <div className="text-xs font-mono font-semibold text-blue-800">06. Gini Impurity Note</div>
            <h3 className="text-sm font-semibold text-slate-900">
              Why Gini Impurity Is Not Used Here
            </h3>
            <p className="text-xs text-slate-600 leading-relaxed">
              In CART, <strong>Gini Impurity</strong> (<span className="font-mono">G = 1 − ∑ pₖ²</span>) measures how often a randomly chosen sample would be misclassified based on class probabilities <span className="font-mono">pₖ</span>. Because car resale price is continuous rather than discrete categories, class probabilities do not exist. Therefore, CART regression replaces Gini impurity with <strong>Squared Error (MSE)</strong> reduction as the mathematically appropriate variance-based impurity measure.
            </p>
          </div>
        </div>
      </div>

      {/* Interactive Mathematical Worked Example */}
      <div className="bg-white border border-slate-200 rounded-xl p-6 space-y-5">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <div className="text-xs font-mono font-semibold text-blue-800">Interactive Worked Math Example</div>
            <h2 className="text-lg font-semibold text-slate-900 mt-0.5">
              How CART Computes the Best Split Using Squared-Error Reduction
            </h2>
            <p className="text-xs text-slate-600 mt-1">
              Test different candidate splits on a 6-car sample (prices in ₹ Lakhs). CART selects whichever split maximizes ΔSSE.
            </p>
          </div>
          <div className="flex flex-wrap gap-1.5">
            {CANDIDATE_SPLITS.map((s) => (
              <button
                key={s.id}
                onClick={() => setSelectedSplitId(s.id)}
                className={`px-3 py-1.5 text-xs font-mono rounded-lg border transition-colors whitespace-nowrap shrink-0 ${
                  selectedSplitId === s.id
                    ? 'bg-blue-800 text-white border-blue-800'
                    : 'bg-white text-slate-700 border-slate-300 hover:bg-slate-50'
                }`}
              >
                {s.label}
              </button>
            ))}
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
          <div className="lg:col-span-5 overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead>
                <tr className="border-b border-slate-200 text-slate-500">
                  <th className="py-1.5 pr-3 font-medium">Car</th>
                  <th className="py-1.5 pr-3 font-medium text-right">Year</th>
                  <th className="py-1.5 pr-3 font-medium text-right">Km</th>
                  <th className="py-1.5 pr-3 font-medium text-right">Price (₹L)</th>
                  <th className="py-1.5 font-medium text-right">Branch</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 font-mono tabular-nums">
                {MINI_DATASET.map((car) => {
                  const isLeft = splitMath.candidate.test(car);
                  return (
                    <tr key={car.car}>
                      <td className="py-1.5 pr-3 font-sans font-medium text-slate-900">{car.car}</td>
                      <td className="py-1.5 pr-3 text-right text-slate-700">{car.year}</td>
                      <td className="py-1.5 pr-3 text-right text-slate-700">{car.km.toLocaleString('en-IN')}</td>
                      <td className="py-1.5 pr-3 text-right text-slate-900">{car.priceLakh.toFixed(2)}</td>
                      <td className="py-1.5 text-right font-semibold text-blue-800">
                        {isLeft ? 'Left' : 'Right'}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          <div className="lg:col-span-7 grid grid-cols-1 sm:grid-cols-3 gap-3 text-xs tabular-nums">
            <div className="bg-slate-50 border border-slate-200 rounded-lg p-3.5 space-y-1">
              <div className="text-slate-500">Parent Node (n=6)</div>
              <div className="font-mono font-semibold text-slate-900">
                Mean ȳ = ₹{splitMath.parentStats.mean.toFixed(2)}L
              </div>
              <div className="font-mono text-slate-700">
                SSE = {splitMath.parentStats.sse.toFixed(3)}
              </div>
              <div className="font-mono text-slate-500">
                MSE = {splitMath.parentStats.mse.toFixed(3)}
              </div>
            </div>

            <div className="bg-blue-50/70 border border-blue-200 rounded-lg p-3.5 space-y-1">
              <div className="text-blue-900 font-medium">Left Child (n={splitMath.left.length})</div>
              <div className="font-mono font-semibold text-slate-900">
                Mean ȳ_L = ₹{splitMath.leftStats.mean.toFixed(2)}L
              </div>
              <div className="font-mono text-slate-700">
                SSE_L = {splitMath.leftStats.sse.toFixed(3)}
              </div>
              <div className="font-mono text-slate-500">
                MSE_L = {splitMath.leftStats.mse.toFixed(3)}
              </div>
            </div>

            <div className="bg-slate-50 border border-slate-200 rounded-lg p-3.5 space-y-1">
              <div className="text-slate-700 font-medium">Right Child (n={splitMath.right.length})</div>
              <div className="font-mono font-semibold text-slate-900">
                Mean ȳ_R = ₹{splitMath.rightStats.mean.toFixed(2)}L
              </div>
              <div className="font-mono text-slate-700">
                SSE_R = {splitMath.rightStats.sse.toFixed(3)}
              </div>
              <div className="font-mono text-slate-500">
                MSE_R = {splitMath.rightStats.mse.toFixed(3)}
              </div>
            </div>

            <div className="sm:col-span-3 bg-slate-900 text-white rounded-lg p-4 flex flex-wrap items-center justify-between gap-3">
              <div className="font-mono text-xs">
                ΔSSE = {splitMath.parentStats.sse.toFixed(3)} − ({splitMath.leftStats.sse.toFixed(3)} + {splitMath.rightStats.sse.toFixed(3)}) ={' '}
                <strong className="text-emerald-300 text-sm">{splitMath.sseReduction.toFixed(3)}</strong>
              </div>
              <div className="text-xs text-slate-300">
                {splitMath.candidate.id === splitMath.bestId
                  ? 'Optimal Split: Highest squared-error reduction among candidates.'
                  : `Tip: Try "${CANDIDATE_SPLITS.find((s) => s.id === splitMath.bestId)?.label}" for maximum reduction (${splitMath.allRanked[0].reduction.toFixed(3)}).`}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Hyperparameters, Overfitting & Pruning */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-white border border-slate-200 rounded-xl p-5 space-y-2">
          <div className="text-xs font-mono font-semibold text-blue-800">07. Structural Controls</div>
          <h3 className="text-sm font-semibold text-slate-900">
            max_depth, min_samples_split & min_samples_leaf
          </h3>
          <ul className="text-xs text-slate-600 space-y-1.5 leading-relaxed list-disc list-inside">
            <li><span className="font-mono font-medium text-slate-800">max_depth</span> caps longest root-to-leaf path, preventing overly complex trees.</li>
            <li><span className="font-mono font-medium text-slate-800">min_samples_split</span> sets minimum samples required before splitting an internal node.</li>
            <li><span className="font-mono font-medium text-slate-800">min_samples_leaf</span> enforces minimum cars per leaf so predictions average multiple vehicles rather than 1 outlier.</li>
          </ul>
        </div>

        <div className="bg-white border border-slate-200 rounded-xl p-5 space-y-2">
          <div className="text-xs font-mono font-semibold text-blue-800">08. Bias–Variance</div>
          <h3 className="text-sm font-semibold text-slate-900">
            Overfitting vs Underfitting
          </h3>
          <p className="text-xs text-slate-600 leading-relaxed">
            An unpruned tree with <span className="font-mono">max_depth=None</span> and <span className="font-mono">min_samples_leaf=1</span> memorizes noise in training cars (Train R² ≈ 1.00, low Test R² — <strong>Overfitting</strong>). Conversely, a shallow tree (<span className="font-mono">max_depth=1</span>) misses key interactions like year and engine size (low Train & Test R² — <strong>Underfitting</strong>).
          </p>
        </div>

        <div className="bg-white border border-slate-200 rounded-xl p-5 space-y-2">
          <div className="text-xs font-mono font-semibold text-blue-800">09. Cost-Complexity</div>
          <h3 className="text-sm font-semibold text-slate-900">
            Pruning (ccp_alpha) & GridSearchCV
          </h3>
          <p className="text-xs text-slate-600 leading-relaxed">
            Minimal Cost-Complexity Pruning minimizes <span className="font-mono">R_α(T) = R(T) + α·|T|</span>, collapsing weak branches whose error reduction does not justify extra leaf nodes <span className="font-mono">|T|</span>. Pairing <span className="font-mono">ccp_alpha</span> with 5-fold <span className="font-mono">GridSearchCV</span> selects the balance that generalizes best to unseen vehicles.
          </p>
        </div>
      </div>

      {/* Live Trained Tree Preview */}
      <div className="space-y-2">
        <div className="text-xs text-slate-600">
          Live Trained CART Structure ({model.treeStats.totalNodes} nodes · {model.treeStats.leafNodes} leaves · Best depth {model.bestParams.max_depth} · Root mean {formatInr(model.tree.prediction)} / {formatLakh(model.tree.prediction)})
        </div>
        <CartTreeDiagram tree={model.tree} maxDisplayDepth={2} />
      </div>
    </div>
  );
}
