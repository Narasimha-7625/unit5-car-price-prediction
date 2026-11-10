import React, { useEffect, useState } from 'react';
import {
  AlertCircle,
  BarChart3,
  BookOpen,
  Calculator,
  Database,
  FileText,
  Home,
  Menu,
  X,
} from 'lucide-react';
import type { CartModelArtifact } from './ml/types.ts';
import { AboutCart } from './pages/AboutCart.tsx';
import { DatasetExplorer } from './pages/DatasetExplorer.tsx';
import { HomeDashboard, type PageId } from './pages/HomeDashboard.tsx';
import { ModelAnalytics } from './pages/ModelAnalytics.tsx';
import { PricePrediction } from './pages/PricePrediction.tsx';
import { ProjectInfo } from './pages/ProjectInfo.tsx';

const NAV_ITEMS: { id: PageId; label: string; icon: React.ComponentType<{ className?: string }> }[] = [
  { id: 'home', label: 'Home Dashboard', icon: Home },
  { id: 'predict', label: 'Price Prediction', icon: Calculator },
  { id: 'dataset', label: 'Dataset Explorer', icon: Database },
  { id: 'analytics', label: 'Model Analytics', icon: BarChart3 },
  { id: 'about', label: 'About CART', icon: BookOpen },
  { id: 'project', label: 'Project Information', icon: FileText },
];

export default function App() {
  const [activePage, setActivePage] = useState<PageId>('home');
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [model, setModel] = useState<CartModelArtifact | null>(null);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);

  const fetchModel = async () => {
    setLoading(true);
    setLoadError(null);
    try {
      const res = await fetch('/api/model');
      const data = await res.json();
      if (!res.ok) {
        setLoadError(data.error ?? 'Failed to load trained CART model.');
        return;
      }
      setModel(data as CartModelArtifact);
    } catch {
      setLoadError('Unable to reach the backend valuation service.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchModel();
  }, []);

  const handleNavigate = (page: PageId) => {
    setActivePage(page);
    setMobileMenuOpen(false);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  const currentNav = NAV_ITEMS.find((n) => n.id === activePage) ?? NAV_ITEMS[0];

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 flex flex-col lg:flex-row">
      {/* Desktop Left Sidebar */}
      <aside className="hidden lg:flex lg:w-64 lg:flex-col lg:fixed lg:inset-y-0 bg-white border-r border-slate-200">
        <div className="px-6 py-5 border-b border-slate-200">
          <div className="font-display text-base font-bold tracking-tight text-slate-900 whitespace-nowrap">
            CART Car Valuation
          </div>
          <div className="text-xs text-slate-500 mt-0.5 whitespace-nowrap">
            Indian Used-Car ML System
          </div>
        </div>

        <nav className="flex-1 px-3 py-4 space-y-1">
          {NAV_ITEMS.map((item) => {
            const Icon = item.icon;
            const isActive = activePage === item.id;
            return (
              <button
                key={item.id}
                onClick={() => handleNavigate(item.id)}
                className={`w-full flex items-center gap-3 px-3 py-2.5 text-sm font-medium rounded-lg transition-colors whitespace-nowrap shrink-0 ${
                  isActive
                    ? 'bg-blue-50 text-blue-900'
                    : 'text-slate-600 hover:bg-slate-100 hover:text-slate-900'
                }`}
              >
                <Icon className={`w-4 h-4 shrink-0 ${isActive ? 'text-blue-800' : 'text-slate-500'}`} />
                <span className="truncate">{item.label}</span>
              </button>
            );
          })}
        </nav>

        <div className="p-4 border-t border-slate-200 text-xs text-slate-500 space-y-1">
          <div className="font-medium text-slate-700">Algorithm Constraint</div>
          <div>DecisionTreeRegressor (CART)</div>
          {model && (
            <div className="font-mono text-[11px] text-slate-500 tabular-nums">
              {model.totalCleanedRows} records · R² {(model.metrics.r2 * 100).toFixed(1)}%
            </div>
          )}
        </div>
      </aside>

      {/* Mobile Drawer */}
      {mobileMenuOpen && (
        <div className="fixed inset-0 z-50 lg:hidden flex">
          <div
            className="fixed inset-0 bg-slate-900/40"
            onClick={() => setMobileMenuOpen(false)}
          />
          <div className="relative w-72 max-w-[80vw] bg-white h-full border-r border-slate-200 flex flex-col">
            <div className="px-5 py-4 border-b border-slate-200 flex items-center justify-between">
              <span className="font-display text-base font-bold text-slate-900">
                CART Car Valuation
              </span>
              <button
                onClick={() => setMobileMenuOpen(false)}
                aria-label="Close navigation menu"
                className="p-1.5 text-slate-500 hover:text-slate-900"
              >
                <X className="w-5 h-5" />
              </button>
            </div>
            <nav className="p-3 space-y-1 flex-1">
              {NAV_ITEMS.map((item) => {
                const Icon = item.icon;
                const isActive = activePage === item.id;
                return (
                  <button
                    key={item.id}
                    onClick={() => handleNavigate(item.id)}
                    className={`w-full flex items-center gap-3 px-3 py-2.5 text-sm font-medium rounded-lg whitespace-nowrap ${
                      isActive
                        ? 'bg-blue-50 text-blue-900'
                        : 'text-slate-600 hover:bg-slate-100'
                    }`}
                  >
                    <Icon className="w-4 h-4 shrink-0" />
                    <span>{item.label}</span>
                  </button>
                );
              })}
            </nav>
          </div>
        </div>
      )}

      {/* Main Content Area */}
      <div className="flex-1 lg:pl-64 flex flex-col min-w-0">
        {/* Top Contextual Header */}
        <header className="sticky top-0 z-30 bg-white/95 backdrop-blur border-b border-slate-200 px-4 sm:px-8 py-3.5 flex items-center justify-between gap-4">
          <div className="flex items-center gap-3 min-w-0">
            <button
              onClick={() => setMobileMenuOpen(true)}
              aria-label="Open navigation menu"
              className="lg:hidden p-2 -ml-2 text-slate-600 hover:text-slate-900 rounded-lg"
            >
              <Menu className="w-5 h-5" />
            </button>
            <div className="flex items-center gap-2 text-sm min-w-0">
              <span className="text-slate-500 hidden sm:inline whitespace-nowrap">
                Used Car Price Prediction
              </span>
              <span className="text-slate-400 hidden sm:inline" aria-hidden="true">
                /
              </span>
              <span className="font-semibold text-slate-900 truncate">
                {currentNav.label}
              </span>
            </div>
          </div>

          <div className="flex items-center gap-3 shrink-0">
            <button
              onClick={() =>
                handleNavigate(activePage === 'predict' ? 'dataset' : 'predict')
              }
              className="px-4 py-2 text-xs font-semibold text-white bg-blue-800 hover:bg-blue-900 rounded-lg transition-colors whitespace-nowrap shrink-0"
            >
              {activePage === 'predict' ? 'Upload CSV Dataset' : 'Predict Car Price'}
            </button>
          </div>
        </header>

        <main className="flex-1 max-w-[1200px] w-full mx-auto px-4 sm:px-8 py-8">
          {loading ? (
            <div className="space-y-6">
              <div className="h-48 bg-white border border-slate-200 rounded-xl animate-pulse" />
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                {[1, 2, 3, 4].map((i) => (
                  <div
                    key={i}
                    className="h-28 bg-white border border-slate-200 rounded-xl animate-pulse"
                  />
                ))}
              </div>
            </div>
          ) : loadError || !model ? (
            <div className="bg-white border border-red-200 rounded-xl p-6 flex flex-col items-start gap-3">
              <div className="flex items-center gap-2 text-red-800 font-semibold text-sm">
                <AlertCircle className="w-4 h-4" />
                <span>Unable to Load CART Model</span>
              </div>
              <p className="text-sm text-slate-600">{loadError}</p>
              <button
                onClick={fetchModel}
                className="px-4 py-2 text-xs font-semibold text-white bg-blue-800 hover:bg-blue-900 rounded-lg"
              >
                Retry Loading Model
              </button>
            </div>
          ) : (
            <>
              {activePage === 'home' && (
                <HomeDashboard model={model} onNavigate={handleNavigate} />
              )}
              {activePage === 'predict' && <PricePrediction model={model} />}
              {activePage === 'dataset' && (
                <DatasetExplorer model={model} onModelReplaced={setModel} />
              )}
              {activePage === 'analytics' && <ModelAnalytics model={model} />}
              {activePage === 'about' && <AboutCart model={model} />}
              {activePage === 'project' && <ProjectInfo model={model} />}
            </>
          )}
        </main>

        <footer className="border-t border-slate-200 bg-white px-4 sm:px-8 py-4 text-xs text-slate-500 flex flex-wrap items-center justify-between gap-2">
          <span>
            Used Car Price Prediction System Using CART Regression (DecisionTreeRegressor)
          </span>
          <span>Academic Machine Learning Project · Indian Used-Car Market</span>
        </footer>
      </div>
    </div>
  );
}
