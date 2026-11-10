import React, { useEffect, useState } from 'react';
import { AlertCircle, Calculator, CheckCircle2, RotateCcw } from 'lucide-react';
import { formatInr } from '../ml/preprocessing.ts';
import type { CartModelArtifact, PredictionResponse } from '../ml/types.ts';

export function PricePrediction({ model }: { model: CartModelArtifact }) {
  const schema = model.preprocessorState.featureSchema;

  const buildDefaults = () => {
    const init: Record<string, string | number> = {};
    for (const field of schema) {
      init[field.key] = field.defaultValue;
    }
    return init;
  };

  const [formValues, setFormValues] = useState<Record<string, string | number>>(buildDefaults);
  const [customModelMode, setCustomModelMode] = useState(false);
  const [fieldErrors, setFieldErrors] = useState<Record<string, string>>({});
  const [serverError, setServerError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);

  useEffect(() => {
    setFormValues(buildDefaults());
    setPrediction(null);
    setFieldErrors({});
    setServerError(null);
  }, [model.trainedAt]);

  const handleFieldChange = (key: string, value: string | number) => {
    setFormValues((prev) => {
      const updated = { ...prev, [key]: value };
      // When brand changes and brand->model hierarchy exists, auto-select first valid model of that brand
      if (key === 'brand' && !customModelMode) {
        const modelField = schema.find((f) => f.key === 'model');
        const brandModels = modelField?.brandModelsMap?.[String(value)];
        if (brandModels && brandModels.length > 0) {
          updated.model = brandModels[0];
        }
      }
      return updated;
    });
    if (fieldErrors[key]) {
      setFieldErrors((prev) => {
        const next = { ...prev };
        delete next[key];
        return next;
      });
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setServerError(null);

    // Client-side pre-validation for immediate feedback
    const localErrors: Record<string, string> = {};
    const currentYear = new Date().getFullYear();
    for (const field of schema) {
      const raw = formValues[field.key];
      if (field.kind === 'categorical') {
        if (!String(raw ?? '').trim()) {
          localErrors[field.key] = `Please enter or select ${field.label.toLowerCase()}.`;
        }
      } else {
        const num = Number(raw);
        if (raw === '' || raw === undefined || !Number.isFinite(num)) {
          localErrors[field.key] = `Please enter a valid numeric ${field.label.toLowerCase()}.`;
        } else if (field.key === 'year' && (!Number.isInteger(num) || num < 1985 || num > currentYear)) {
          localErrors[field.key] = `Year must be between 1985 and ${currentYear}.`;
        } else if (field.key === 'km_driven' && (num < 0 || num > 1500000)) {
          localErrors[field.key] = 'Kilometres driven cannot be negative or exceed 15,00,000 km.';
        } else if (field.key === 'seats' && (!Number.isInteger(num) || num < 2 || num > 12)) {
          localErrors[field.key] = 'Seats must be an integer between 2 and 12.';
        } else if (num < 0) {
          localErrors[field.key] = `${field.label} cannot be negative.`;
        }
      }
    }

    if (Object.keys(localErrors).length > 0) {
      setFieldErrors(localErrors);
      return;
    }

    setLoading(true);
    try {
      const res = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formValues),
      });
      const data = await res.json();
      if (!res.ok) {
        if (data.fieldErrors) setFieldErrors(data.fieldErrors);
        setServerError(data.error ?? 'Prediction request failed.');
        return;
      }
      setFieldErrors({});
      setPrediction(data as PredictionResponse);
    } catch {
      setServerError('Network error while contacting the prediction service.');
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setFormValues(buildDefaults());
    setFieldErrors({});
    setServerError(null);
    setPrediction(null);
    setCustomModelMode(false);
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-slate-900">
          Interactive Used Car Price Prediction
        </h1>
        <p className="text-sm text-slate-600 mt-1">
          Input fields automatically adapt to the {schema.length} specification features supported by the currently saved CART pipeline ({model.datasetName}).
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* Left: Input Form */}
        <form
          onSubmit={handleSubmit}
          className="lg:col-span-7 bg-white border border-slate-200 rounded-xl p-6 space-y-5"
        >
          <div className="flex items-center justify-between border-b border-slate-100 pb-3">
            <h2 className="text-base font-semibold text-slate-900">
              Vehicle Specifications
            </h2>
            <button
              type="button"
              onClick={handleReset}
              className="inline-flex items-center gap-1.5 text-xs font-medium text-slate-600 hover:text-slate-900 whitespace-nowrap shrink-0"
            >
              <RotateCcw className="w-3.5 h-3.5" />
              Reset Defaults
            </button>
          </div>

          {serverError && (
            <div className="p-3 bg-red-50 border border-red-200 rounded-lg flex items-start gap-2.5 text-xs text-red-800">
              <AlertCircle className="w-4 h-4 shrink-0 mt-0.5 text-red-600" />
              <span>{serverError}</span>
            </div>
          )}

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            {schema.map((field) => {
              const err = fieldErrors[field.key];
              const isModelField = field.key === 'model';
              const selectedBrand = String(formValues.brand ?? '');
              const brandModels =
                isModelField && field.brandModelsMap && selectedBrand
                  ? field.brandModelsMap[selectedBrand]
                  : undefined;
              const selectOptions = brandModels ?? field.options ?? [];

              return (
                <div key={field.key} className="space-y-1.5">
                  <div className="flex items-center justify-between gap-2">
                    <label
                      htmlFor={`field-${field.key}`}
                      className="block text-xs font-medium text-slate-700"
                    >
                      {field.label}
                      {field.unit ? ` (${field.unit})` : ''}
                    </label>
                    {isModelField && (
                      <button
                        type="button"
                        onClick={() => setCustomModelMode((v) => !v)}
                        className="text-[11px] text-blue-800 hover:underline whitespace-nowrap shrink-0"
                      >
                        {customModelMode ? 'Use dropdown' : 'Type custom model'}
                      </button>
                    )}
                  </div>

                  {field.kind === 'categorical' && !(isModelField && customModelMode) ? (
                    <select
                      id={`field-${field.key}`}
                      value={String(formValues[field.key] ?? '')}
                      onChange={(e) => handleFieldChange(field.key, e.target.value)}
                      className={`w-full px-3 py-2 text-sm bg-white border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-800/20 ${
                        err ? 'border-red-400' : 'border-slate-300'
                      }`}
                    >
                      {selectOptions.map((opt) => (
                        <option key={opt} value={opt}>
                          {opt}
                        </option>
                      ))}
                    </select>
                  ) : field.kind === 'categorical' ? (
                    <input
                      id={`field-${field.key}`}
                      type="text"
                      value={String(formValues[field.key] ?? '')}
                      onChange={(e) => handleFieldChange(field.key, e.target.value)}
                      placeholder="Enter car model (unseen categories handled safely)"
                      className={`w-full px-3 py-2 text-sm bg-white border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-800/20 ${
                        err ? 'border-red-400' : 'border-slate-300'
                      }`}
                    />
                  ) : (
                    <input
                      id={`field-${field.key}`}
                      type="number"
                      step={field.step ?? 'any'}
                      value={formValues[field.key] ?? ''}
                      onChange={(e) => handleFieldChange(field.key, e.target.value)}
                      className={`w-full px-3 py-2 text-sm font-mono tabular-nums bg-white border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-800/20 ${
                        err ? 'border-red-400' : 'border-slate-300'
                      }`}
                    />
                  )}

                  {err && <p className="text-xs text-red-600">{err}</p>}
                </div>
              );
            })}
          </div>

          <div className="pt-2 flex flex-wrap items-center justify-between gap-4 border-t border-slate-100">
            <span className="text-xs text-slate-500">
              Uses persisted CART pipeline without retraining on refresh.
            </span>
            <button
              type="submit"
              disabled={loading}
              className="inline-flex items-center gap-2 px-6 py-2.5 text-sm font-semibold text-white bg-blue-800 hover:bg-blue-900 disabled:opacity-60 rounded-lg transition-colors whitespace-nowrap shrink-0"
            >
              <Calculator className="w-4 h-4" />
              {loading ? 'Evaluating Tree...' : 'Predict Price'}
            </button>
          </div>
        </form>

        {/* Right: Valuation Result & Decision Path */}
        <div className="lg:col-span-5 space-y-5">
          {prediction ? (
            <>
              <div className="bg-white border border-slate-200 rounded-xl p-6 space-y-4">
                <div className="flex items-center gap-2 text-xs font-medium text-emerald-700">
                  <CheckCircle2 className="w-4 h-4" />
                  <span>Estimated Resale Valuation (CART Leaf Mean)</span>
                </div>

                <div>
                  <div className="text-3xl sm:text-4xl font-bold text-slate-900 font-mono tabular-nums">
                    {prediction.formattedInr}
                  </div>
                  <div className="text-sm font-medium text-blue-800 font-mono tabular-nums mt-1">
                    {prediction.formattedLakh}
                  </div>
                </div>

                <div className="p-3 bg-amber-50 border border-amber-200 rounded-lg text-xs text-amber-900 leading-relaxed">
                  <strong>Estimate Notice:</strong> This is a machine learning valuation based on {prediction.leafSamples} similar training vehicle(s) in the terminal CART leaf node, not a guaranteed selling price. Actual market offers depend on physical inspection, insurance history, and regional demand.
                </div>

                <div className="pt-3 border-t border-slate-100">
                  <div className="text-xs font-semibold text-slate-700 mb-2">
                    Entered Vehicle Summary
                  </div>
                  <dl className="grid grid-cols-2 gap-x-4 gap-y-2 text-xs">
                    {Object.entries(prediction.validatedInputs).map(([k, v]) => {
                      const label = schema.find((f) => f.key === k)?.label ?? k;
                      return (
                        <div key={k} className="flex justify-between gap-2 border-b border-slate-100 pb-1">
                          <dt className="text-slate-500 truncate">{label}</dt>
                          <dd className="font-mono font-medium text-slate-900 tabular-nums truncate">
                            {typeof v === 'number' && k === 'km_driven'
                              ? `${v.toLocaleString('en-IN')} km`
                              : String(v)}
                          </dd>
                        </div>
                      );
                    })}
                  </dl>
                </div>
              </div>

              <div className="bg-white border border-slate-200 rounded-xl p-6 space-y-3">
                <div>
                  <h3 className="text-sm font-semibold text-slate-900">
                    CART Decision Path Trace ({prediction.decisionPath.length} splits)
                  </h3>
                  <p className="text-xs text-slate-500 mt-0.5">
                    Exact binary thresholds traversed from root node to leaf prediction
                  </p>
                </div>

                <ol className="space-y-2">
                  {prediction.decisionPath.map((step, i) => (
                    <li
                      key={`${step.nodeId}-${i}`}
                      className="text-xs bg-slate-50 border border-slate-200 rounded-lg p-2.5 flex items-center justify-between gap-3"
                    >
                      <div className="space-y-0.5 min-w-0">
                        <div className="font-mono font-medium text-slate-900 truncate">
                          Depth {step.depth}: {step.ruleText}
                        </div>
                        <div className="text-slate-500 tabular-nums">
                          {step.samples} training cars in node · Node avg {formatInr(step.nodeMeanPrice)}
                        </div>
                      </div>
                      <span className="font-mono text-[11px] font-semibold text-blue-800 whitespace-nowrap shrink-0">
                        {step.direction === 'left' ? 'True (Left)' : 'False (Right)'}
                      </span>
                    </li>
                  ))}
                </ol>
              </div>
            </>
          ) : (
            <div className="bg-white border border-slate-200 rounded-xl p-6 space-y-3">
              <h3 className="text-base font-semibold text-slate-900">
                Ready for Valuation
              </h3>
              <p className="text-sm text-slate-600 leading-relaxed">
                Adjust the vehicle specifications on the left and click <strong>Predict Price</strong>. The system validates every input, applies the fitted preprocessing transformations, and evaluates the saved CART DecisionTreeRegressor.
              </p>
              <div className="pt-2 border-t border-slate-100 text-xs text-slate-500 space-y-1">
                <div>Supported active features: {schema.map((s) => s.label).join(', ')}</div>
                <div>Unseen model names or rare categories are safely mapped without errors.</div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
