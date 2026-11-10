import React, { useMemo, useState } from 'react';
import {
  AlertCircle,
  CheckCircle2,
  Download,
  ExternalLink,
  RefreshCw,
  Search,
  Upload,
} from 'lucide-react';
import {
  CategoryComparisonChart,
  HistogramChart,
  ScatterPlotChart,
} from '../components/Charts.tsx';
import { formatInr, formatLakh } from '../ml/preprocessing.ts';
import type { CartModelArtifact } from '../ml/types.ts';

export function DatasetExplorer({
  model,
  onModelReplaced,
}: {
  model: CartModelArtifact;
  onModelReplaced: (nextModel: CartModelArtifact) => void;
}) {
  const [selectedFile, setSelectedFile] = useState<{ name: string; text: string } | null>(null);
  const [statusMessage, setStatusMessage] = useState<string | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [showGuide, setShowGuide] = useState(false);

  const handleFilePick = async (e: React.ChangeEvent<HTMLInputElement>) => {
    setErrorMessage(null);
    setStatusMessage(null);
    const file = e.target.files?.[0];
    if (!file) return;
    if (!file.name.toLowerCase().endsWith('.csv')) {
      setErrorMessage('Please select a valid .csv file.');
      return;
    }
    const text = await file.text();
    setSelectedFile({ name: file.name, text });
    setStatusMessage(`Loaded "${file.name}" (${(file.size / 1024).toFixed(1)} KB). Click "Replace Dataset & Retrain CART Model" to train.`);
  };

  const handleUploadAndRetrain = async () => {
    if (!selectedFile) return;
    setBusy(true);
    setErrorMessage(null);
    setStatusMessage(null);
    try {
      const res = await fetch('/api/upload-retrain', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          csvText: selectedFile.text,
          fileName: selectedFile.name,
        }),
      });
      const data = await res.json();
      if (!res.ok) {
        setErrorMessage(data.error ?? 'Failed to preprocess CSV or retrain CART model.');
        return;
      }
      onModelReplaced(data as CartModelArtifact);
      setSelectedFile(null);
      setStatusMessage(
        `Successfully cleaned dataset and retrained CART model on ${(data as CartModelArtifact).totalCleanedRows} valid records.`
      );
    } catch {
      setErrorMessage('Network error while uploading dataset.');
    } finally {
      setBusy(false);
    }
  };

  const handleRestoreSample = async () => {
    setBusy(true);
    setErrorMessage(null);
    setStatusMessage(null);
    try {
      const res = await fetch('/api/restore-sample', { method: 'POST' });
      const data = await res.json();
      if (!res.ok) {
        setErrorMessage(data.error ?? 'Failed to restore sample dataset.');
        return;
      }
      onModelReplaced(data as CartModelArtifact);
      setStatusMessage('Restored built-in Sample Demonstration Dataset and retrained CART model.');
    } catch {
      setErrorMessage('Network error while restoring sample dataset.');
    } finally {
      setBusy(false);
    }
  };

  const filteredPreview = useMemo(() => {
    const q = searchQuery.trim().toLowerCase();
    if (!q) return model.previewRows.slice(0, 25);
    return model.previewRows
      .filter((row) =>
        Object.values(row).some((val) => String(val ?? '').toLowerCase().includes(q))
      )
      .slice(0, 25);
  }, [model.previewRows, searchQuery]);

  const previewColumns = useMemo(() => {
    const base = ['brand', 'model', 'year', 'selling_price', 'km_driven', 'fuel', 'transmission', 'owner', 'mileage', 'engine', 'max_power', 'seats'];
    return base.filter((col) =>
      model.previewRows.some((r) => r[col] !== undefined)
    );
  }, [model.previewRows]);

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-slate-900">
            Dataset Explorer & Preprocessing Pipeline
          </h1>
          <p className="text-sm text-slate-600 mt-1">
            Inspect active dataset dimensions, missing-value handling, CSV upload replacement, and Indian market price distributions.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <a
            href="/api/sample-template"
            download="indian_used_cars_template.csv"
            className="inline-flex items-center gap-1.5 px-3.5 py-2 text-xs font-medium text-slate-700 bg-white border border-slate-300 hover:bg-slate-50 rounded-lg transition-colors whitespace-nowrap shrink-0"
          >
            <Download className="w-3.5 h-3.5" />
            Download CSV Template
          </a>
          <button
            type="button"
            onClick={() => setShowGuide((v) => !v)}
            className="inline-flex items-center gap-1.5 px-3.5 py-2 text-xs font-medium text-blue-800 bg-blue-50 border border-blue-200 hover:bg-blue-100 rounded-lg transition-colors whitespace-nowrap shrink-0"
          >
            <ExternalLink className="w-3.5 h-3.5" />
            {showGuide ? 'Hide Kaggle Guide' : 'Kaggle Dataset Guide'}
          </button>
        </div>
      </div>

      {showGuide && (
        <div className="bg-white border border-slate-200 rounded-xl p-5 space-y-3 text-xs text-slate-700">
          <h2 className="text-sm font-semibold text-slate-900">
            How to Download & Use a Real Indian Used-Car Kaggle Dataset
          </h2>
          <ol className="list-decimal list-inside space-y-1.5 leading-relaxed">
            <li>
              Search Kaggle for <strong>"Car details v3.csv" (Vehicle Dataset from CarDekho)</strong> or <strong>"Used Cars Dataset India"</strong> and download the CSV file to your computer.
            </li>
            <li>
              Supported columns include <code className="font-mono bg-slate-100 px-1 rounded">selling_price</code> (or <code className="font-mono bg-slate-100 px-1 rounded">price</code>), <code className="font-mono bg-slate-100 px-1 rounded">name</code> (auto-split into brand & model) or <code className="font-mono bg-slate-100 px-1 rounded">brand</code>/<code className="font-mono bg-slate-100 px-1 rounded">model</code>, <code className="font-mono bg-slate-100 px-1 rounded">year</code>, <code className="font-mono bg-slate-100 px-1 rounded">km_driven</code>, <code className="font-mono bg-slate-100 px-1 rounded">fuel</code>, <code className="font-mono bg-slate-100 px-1 rounded">transmission</code>, <code className="font-mono bg-slate-100 px-1 rounded">owner</code>, <code className="font-mono bg-slate-100 px-1 rounded">mileage</code>, <code className="font-mono bg-slate-100 px-1 rounded">engine</code>, <code className="font-mono bg-slate-100 px-1 rounded">max_power</code>, and <code className="font-mono bg-slate-100 px-1 rounded">seats</code>.
            </li>
            <li>
              Upload the CSV below and click <strong>Replace Dataset & Retrain CART Model</strong>. Units such as "kmpl", "CC", "bhp", "₹", and "Lakh" are parsed automatically.
            </li>
          </ol>
        </div>
      )}

      {/* Upload & Retrain Control Panel */}
      <div className="bg-white border border-slate-200 rounded-xl p-5 space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div className="space-y-1">
            <div className="text-xs text-slate-500">
              Active Dataset ({model.datasetSource === 'sample_demo' ? 'Sample Demonstration Data' : 'User Uploaded Real CSV'})
            </div>
            <div className="text-sm font-semibold text-slate-900">
              {model.datasetName}
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-3">
            <label className="inline-flex items-center gap-2 px-4 py-2 text-xs font-medium text-slate-700 bg-slate-50 border border-slate-300 hover:bg-slate-100 rounded-lg cursor-pointer whitespace-nowrap shrink-0">
              <Upload className="w-3.5 h-3.5" />
              <span>Select CSV File</span>
              <input
                type="file"
                accept=".csv,text/csv"
                onChange={handleFilePick}
                className="sr-only"
              />
            </label>

            <button
              type="button"
              disabled={!selectedFile || busy}
              onClick={handleUploadAndRetrain}
              className="inline-flex items-center gap-2 px-4 py-2 text-xs font-semibold text-white bg-blue-800 hover:bg-blue-900 disabled:opacity-50 rounded-lg transition-colors whitespace-nowrap shrink-0"
            >
              <RefreshCw className={`w-3.5 h-3.5 ${busy ? 'animate-spin' : ''}`} />
              {busy ? 'Training CART Model...' : 'Replace Dataset & Retrain CART Model'}
            </button>

            {model.datasetSource === 'uploaded_csv' && (
              <button
                type="button"
                disabled={busy}
                onClick={handleRestoreSample}
                className="px-3.5 py-2 text-xs font-medium text-slate-700 bg-white border border-slate-300 hover:bg-slate-50 rounded-lg whitespace-nowrap shrink-0"
              >
                Restore Sample Dataset
              </button>
            )}
          </div>
        </div>

        {errorMessage && (
          <div className="p-3 bg-red-50 border border-red-200 rounded-lg flex items-start gap-2 text-xs text-red-800">
            <AlertCircle className="w-4 h-4 shrink-0 mt-0.5 text-red-600" />
            <span>{errorMessage}</span>
          </div>
        )}
        {statusMessage && (
          <div className="p-3 bg-emerald-50 border border-emerald-200 rounded-lg flex items-start gap-2 text-xs text-emerald-900">
            <CheckCircle2 className="w-4 h-4 shrink-0 mt-0.5 text-emerald-700" />
            <span>{statusMessage}</span>
          </div>
        )}
      </div>

      {/* Preprocessing Summary Metrics */}
      <div className="grid grid-cols-2 lg:grid-cols-5 gap-4">
        <div className="bg-white border border-slate-200 rounded-xl p-4">
          <div className="text-xs text-slate-500">Raw Dimensions</div>
          <div className="mt-1 text-lg font-bold font-mono tabular-nums text-slate-900">
            {model.preprocessing.rawRows} × {model.preprocessing.rawColumns}
          </div>
        </div>
        <div className="bg-white border border-slate-200 rounded-xl p-4">
          <div className="text-xs text-slate-500">Cleaned Records</div>
          <div className="mt-1 text-lg font-bold font-mono tabular-nums text-slate-900">
            {model.preprocessing.cleanedRows}
          </div>
        </div>
        <div className="bg-white border border-slate-200 rounded-xl p-4">
          <div className="text-xs text-slate-500">Duplicates Removed</div>
          <div className="mt-1 text-lg font-bold font-mono tabular-nums text-slate-900">
            {model.preprocessing.duplicatesRemoved}
          </div>
        </div>
        <div className="bg-white border border-slate-200 rounded-xl p-4">
          <div className="text-xs text-slate-500">Invalid Rows Filtered</div>
          <div className="mt-1 text-lg font-bold font-mono tabular-nums text-slate-900">
            {model.preprocessing.invalidRowsRemoved}
          </div>
        </div>
        <div className="bg-white border border-slate-200 rounded-xl p-4">
          <div className="text-xs text-slate-500">Missing Values Imputed</div>
          <div className="mt-1 text-lg font-bold font-mono tabular-nums text-slate-900">
            {model.preprocessing.missingValuesTotal}
          </div>
        </div>
      </div>

      {/* Column Schema & Missing Values Table + Preprocessing Steps */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        <div className="lg:col-span-8 bg-white border border-slate-200 rounded-xl p-5 space-y-3">
          <h2 className="text-base font-semibold text-slate-900">
            Column Types, Missing Values & Feature Mapping
          </h2>
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead>
                <tr className="border-b border-slate-200 text-slate-500">
                  <th className="py-2 pr-3 font-medium">Column</th>
                  <th className="py-2 pr-3 font-medium">Role</th>
                  <th className="py-2 pr-3 font-medium">Type</th>
                  <th className="py-2 pr-3 font-medium text-right">Missing</th>
                  <th className="py-2 pr-3 font-medium text-right">Unique</th>
                  <th className="py-2 font-medium">Samples</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {model.preprocessing.columnProfiles.map((col) => (
                  <tr key={col.name} className="hover:bg-slate-50">
                    <td className="py-2 pr-3 font-mono font-medium text-slate-900">{col.name}</td>
                    <td className="py-2 pr-3 font-mono text-slate-700">{col.mappedRole}</td>
                    <td className="py-2 pr-3 text-slate-600 capitalize">{col.inferredType}</td>
                    <td className="py-2 pr-3 font-mono tabular-nums text-right text-slate-700">
                      {col.missingCount}
                    </td>
                    <td className="py-2 pr-3 font-mono tabular-nums text-right text-slate-700">
                      {col.uniqueCount}
                    </td>
                    <td className="py-2 text-slate-500 truncate max-w-[200px]">
                      {col.sampleValues.join(', ')}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        <div className="lg:col-span-4 bg-white border border-slate-200 rounded-xl p-5 space-y-3">
          <h2 className="text-base font-semibold text-slate-900">
            Preprocessing Audit Log
          </h2>
          <ul className="space-y-2 text-xs text-slate-600 leading-relaxed list-disc list-inside">
            {model.preprocessing.stepsApplied.map((step, idx) => (
              <li key={idx}>{step}</li>
            ))}
          </ul>
          {model.preprocessing.invalidReasons.length > 0 && (
            <div className="pt-3 border-t border-slate-100 space-y-1.5">
              <div className="text-xs font-semibold text-slate-800">Invalid Records Breakdown</div>
              {model.preprocessing.invalidReasons.map((r) => (
                <div key={r.reason} className="flex justify-between text-xs text-slate-600">
                  <span>{r.reason}</span>
                  <span className="font-mono tabular-nums font-medium">{r.count}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Actual Dataset Visualizations */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <HistogramChart
          title="Used Car Price Distribution"
          subtitle="Frequency of cleaned vehicle records across Indian Rupee price brackets"
          bins={model.charts.priceHistogram}
          xAxisLabel="Selling Price Range (INR in Lakhs)"
        />
        <CategoryComparisonChart
          title="Price Distribution by Fuel Type"
          subtitle="Minimum, maximum span, average marker, and median price per fuel category"
          items={model.charts.priceByFuel.map((f) => ({
            label: f.fuel,
            count: f.count,
            avgPrice: f.avgPrice,
            medianPrice: f.medianPrice,
            minPrice: f.minPrice,
            maxPrice: f.maxPrice,
          }))}
        />
        <CategoryComparisonChart
          title="Price Distribution by Transmission"
          subtitle="Resale price comparison between Manual and Automatic vehicles"
          items={model.charts.priceByTransmission.map((t) => ({
            label: t.transmission,
            count: t.count,
            avgPrice: t.avgPrice,
            medianPrice: t.medianPrice,
            minPrice: t.minPrice,
            maxPrice: t.maxPrice,
          }))}
        />
        <ScatterPlotChart
          title="Manufacturing Year vs Selling Price"
          subtitle="Newer manufacturing years correlate with higher retained resale value"
          points={model.charts.yearVsPrice.map((p) => ({
            x: p.year,
            y: p.price,
            label: p.label,
            group: p.fuel,
          }))}
          xLabel="Manufacturing Year"
          yLabel="Selling Price"
          formatX={(v) => String(Math.round(v))}
          formatY={(v) =>
            Math.abs(v) >= 10000000 ? `${(v / 10000000).toFixed(1)}Cr` : `${(v / 100000).toFixed(1)}L`
          }
        />
        <div className="lg:col-span-2">
          <ScatterPlotChart
            title="Kilometres Driven vs Selling Price"
            subtitle="Higher odometer readings generally correspond to lower resale prices"
            points={model.charts.kmVsPrice.map((p) => ({
              x: p.km,
              y: p.price,
              label: p.label,
              group: p.transmission,
            }))}
            xLabel="Kilometres Driven"
            yLabel="Selling Price"
            formatX={(v) => `${Math.round(v / 1000)}k`}
            formatY={(v) =>
              Math.abs(v) >= 10000000 ? `${(v / 10000000).toFixed(1)}Cr` : `${(v / 100000).toFixed(1)}L`
            }
          />
        </div>
      </div>

      {/* Searchable Cleaned Dataset Preview */}
      <div className="bg-white border border-slate-200 rounded-xl p-5 space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h2 className="text-base font-semibold text-slate-900">
              Cleaned Dataset Preview (Showing up to 25 of {model.totalCleanedRows} records)
            </h2>
            <p className="text-xs text-slate-500 mt-0.5">
              All currency strings and unit columns normalized to numerical INR and standard units
            </p>
          </div>
          <div className="relative w-full sm:w-64">
            <Search className="w-3.5 h-3.5 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2" />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Filter brand, model, fuel..."
              className="w-full pl-8 pr-3 py-1.5 text-xs bg-slate-50 border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-800/20"
            />
          </div>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead>
              <tr className="border-b border-slate-200 text-slate-500">
                {previewColumns.map((col) => (
                  <th key={col} className="py-2 pr-4 font-medium capitalize whitespace-nowrap">
                    {col.replace('_', ' ')}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {filteredPreview.map((row, idx) => (
                <tr key={idx} className="hover:bg-slate-50">
                  {previewColumns.map((col) => {
                    const val = row[col];
                    const isNum = typeof val === 'number';
                    return (
                      <td
                        key={col}
                        className={`py-2 pr-4 whitespace-nowrap ${
                          isNum ? 'font-mono tabular-nums text-slate-800' : 'text-slate-700'
                        }`}
                      >
                        {col === 'selling_price' && typeof val === 'number'
                          ? `${formatInr(val)} (${formatLakh(val)})`
                          : col === 'km_driven' && typeof val === 'number'
                          ? val.toLocaleString('en-IN')
                          : val ?? 'Imputed'}
                      </td>
                    );
                  })}
                </tr>
              ))}
              {filteredPreview.length === 0 && (
                <tr>
                  <td colSpan={previewColumns.length} className="py-6 text-center text-slate-500">
                    No records match "{searchQuery}".
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
