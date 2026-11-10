import React, { useMemo, useState } from 'react';
import { formatInr, formatLakh } from '../ml/preprocessing.ts';
import type {
  ActualVsPredictedPoint,
  CartTreeNode,
  FeatureImportanceItem,
} from '../ml/types.ts';

interface HistogramBin {
  rangeLabel: string;
  min: number;
  max: number;
  count: number;
}

export function HistogramChart({
  title,
  subtitle,
  bins,
  barColor = '#1E40AF',
  xAxisLabel,
}: {
  title: string;
  subtitle: string;
  bins: HistogramBin[];
  barColor?: string;
  xAxisLabel: string;
}) {
  const [hoveredIdx, setHoveredIdx] = useState<number | null>(null);
  const maxCount = Math.max(1, ...bins.map((b) => b.count));
  const width = 640;
  const height = 270;
  const padL = 46;
  const padR = 16;
  const padT = 20;
  const padB = 54;
  const plotW = width - padL - padR;
  const plotH = height - padT - padB;
  const barGap = 8;
  const barW = bins.length > 0 ? (plotW - barGap * (bins.length - 1)) / bins.length : 40;

  return (
    <div className="bg-white border border-slate-200 rounded-xl p-5">
      <div className="mb-3">
        <h3 className="text-base font-semibold text-slate-900">{title}</h3>
        <p className="text-xs text-slate-500 mt-0.5">{subtitle}</p>
      </div>
      <div className="relative">
        <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-auto overflow-visible">
          {[0, 0.25, 0.5, 0.75, 1].map((t) => {
            const y = padT + plotH * (1 - t);
            const val = Math.round(maxCount * t);
            return (
              <g key={t}>
                <line x1={padL} y1={y} x2={width - padR} y2={y} stroke="#E2E8F0" strokeDasharray="3 3" />
                <text
                  x={padL - 8}
                  y={y + 4}
                  textAnchor="end"
                  className="fill-slate-500 text-[11px] font-mono tabular-nums"
                >
                  {val}
                </text>
              </g>
            );
          })}
          {bins.map((bin, idx) => {
            const h = (bin.count / maxCount) * plotH;
            const x = padL + idx * (barW + barGap);
            const y = padT + plotH - h;
            const isHovered = hoveredIdx === idx;
            return (
              <g
                key={bin.rangeLabel}
                onMouseEnter={() => setHoveredIdx(idx)}
                onMouseLeave={() => setHoveredIdx(null)}
                className="cursor-pointer"
              >
                <rect
                  x={x}
                  y={y}
                  width={Math.max(4, barW)}
                  height={Math.max(2, h)}
                  rx={4}
                  fill={barColor}
                  opacity={hoveredIdx === null || isHovered ? 0.92 : 0.45}
                />
                <text
                  x={x + barW / 2}
                  y={y - 6}
                  textAnchor="middle"
                  className="fill-slate-700 text-[11px] font-mono tabular-nums font-medium"
                >
                  {bin.count}
                </text>
                <text
                  x={x + barW / 2}
                  y={padT + plotH + 18}
                  textAnchor="middle"
                  className="fill-slate-600 text-[10px] font-mono"
                >
                  {bin.rangeLabel}
                </text>
              </g>
            );
          })}
          <text
            x={padL + plotW / 2}
            y={height - 6}
            textAnchor="middle"
            className="fill-slate-500 text-[11px]"
          >
            {xAxisLabel}
          </text>
        </svg>
        {hoveredIdx !== null && bins[hoveredIdx] && (
          <div className="mt-2 text-xs text-slate-600 bg-slate-50 border border-slate-200 rounded-lg px-3 py-1.5 flex items-center justify-between tabular-nums">
            <span>Range: {bins[hoveredIdx].rangeLabel}</span>
            <span className="font-mono font-medium text-slate-900">
              {bins[hoveredIdx].count} vehicles
            </span>
          </div>
        )}
      </div>
    </div>
  );
}

export function CategoryComparisonChart({
  title,
  subtitle,
  items,
}: {
  title: string;
  subtitle: string;
  items: {
    label: string;
    count: number;
    avgPrice: number;
    medianPrice: number;
    minPrice: number;
    maxPrice: number;
  }[];
}) {
  const maxVal = Math.max(1, ...items.map((i) => i.maxPrice));

  return (
    <div className="bg-white border border-slate-200 rounded-xl p-5">
      <div className="mb-4">
        <h3 className="text-base font-semibold text-slate-900">{title}</h3>
        <p className="text-xs text-slate-500 mt-0.5">{subtitle}</p>
      </div>
      <div className="space-y-4">
        {items.map((item) => {
          const avgPct = (item.avgPrice / maxVal) * 100;
          const minPct = (item.minPrice / maxVal) * 100;
          const spanPct = Math.max(2, ((item.maxPrice - item.minPrice) / maxVal) * 100);
          return (
            <div key={item.label} className="space-y-1.5">
              <div className="flex items-center justify-between text-sm">
                <div className="flex items-center gap-2">
                  <span className="font-medium text-slate-900">{item.label}</span>
                  <span className="text-xs text-slate-500 tabular-nums">
                    ({item.count} cars)
                  </span>
                </div>
                <div className="text-xs font-mono tabular-nums text-slate-700">
                  Avg {formatLakh(item.avgPrice)} · Median {formatLakh(item.medianPrice)}
                </div>
              </div>
              <div className="relative h-6 bg-slate-100 rounded-md overflow-hidden border border-slate-200">
                <div
                  className="absolute top-1.5 bottom-1.5 bg-blue-200 rounded"
                  style={{ left: `${minPct}%`, width: `${spanPct}%` }}
                  title={`Min ${formatInr(item.minPrice)} – Max ${formatInr(item.maxPrice)}`}
                />
                <div
                  className="absolute top-0 bottom-0 w-1.5 bg-blue-800 rounded-sm"
                  style={{ left: `calc(${Math.min(98, avgPct)}% - 3px)` }}
                  title={`Average: ${formatInr(item.avgPrice)}`}
                />
              </div>
              <div className="flex justify-between text-[11px] font-mono tabular-nums text-slate-500">
                <span>Min {formatLakh(item.minPrice)}</span>
                <span>Max {formatLakh(item.maxPrice)}</span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

export function ScatterPlotChart({
  title,
  subtitle,
  points,
  xLabel,
  yLabel,
  formatX,
  formatY,
  showDiagonal = false,
}: {
  title: string;
  subtitle: string;
  points: { x: number; y: number; label: string; group: string }[];
  xLabel: string;
  yLabel: string;
  formatX: (v: number) => string;
  formatY: (v: number) => string;
  showDiagonal?: boolean;
}) {
  const [activePoint, setActivePoint] = useState<{
    x: number;
    y: number;
    label: string;
    group: string;
  } | null>(null);

  const width = 640;
  const height = 300;
  const padL = 62;
  const padR = 20;
  const padT = 18;
  const padB = 46;
  const plotW = width - padL - padR;
  const plotH = height - padT - padB;

  const { minX, maxX, minY, maxY } = useMemo(() => {
    if (points.length === 0) return { minX: 0, maxX: 1, minY: 0, maxY: 1 };
    const xs = points.map((p) => p.x);
    const ys = points.map((p) => p.y);
    let loX = Math.min(...xs);
    let hiX = Math.max(...xs);
    let loY = Math.min(...ys);
    let hiY = Math.max(...ys);
    if (showDiagonal) {
      const lo = Math.min(loX, loY);
      const hi = Math.max(hiX, hiY);
      loX = lo;
      loY = lo;
      hiX = hi;
      hiY = hi;
    }
    const spanX = Math.max(1, hiX - loX);
    const spanY = Math.max(1, hiY - loY);
    return {
      minX: loX - spanX * 0.04,
      maxX: hiX + spanX * 0.04,
      minY: loY - spanY * 0.04,
      maxY: hiY + spanY * 0.06,
    };
  }, [points, showDiagonal]);

  const toPxX = (v: number) => padL + ((v - minX) / Math.max(1e-9, maxX - minX)) * plotW;
  const toPxY = (v: number) => padT + plotH - ((v - minY) / Math.max(1e-9, maxY - minY)) * plotH;

  return (
    <div className="bg-white border border-slate-200 rounded-xl p-5">
      <div className="mb-3 flex flex-wrap items-start justify-between gap-2">
        <div>
          <h3 className="text-base font-semibold text-slate-900">{title}</h3>
          <p className="text-xs text-slate-500 mt-0.5">{subtitle}</p>
        </div>
        {showDiagonal && (
          <span className="text-xs font-mono text-slate-600">
            Dashed line: Ideal fit (Predicted = Actual)
          </span>
        )}
      </div>

      <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-auto overflow-visible">
        {[0, 0.25, 0.5, 0.75, 1].map((t) => {
          const yVal = minY + (maxY - minY) * t;
          const yPx = toPxY(yVal);
          const xVal = minX + (maxX - minX) * t;
          const xPx = toPxX(xVal);
          return (
            <g key={t}>
              <line x1={padL} y1={yPx} x2={width - padR} y2={yPx} stroke="#E2E8F0" strokeDasharray="3 3" />
              <text
                x={padL - 8}
                y={yPx + 4}
                textAnchor="end"
                className="fill-slate-500 text-[10px] font-mono tabular-nums"
              >
                {formatY(yVal)}
              </text>
              <line x1={xPx} y1={padT} x2={xPx} y2={padT + plotH} stroke="#F1F5F9" />
              <text
                x={xPx}
                y={padT + plotH + 16}
                textAnchor="middle"
                className="fill-slate-500 text-[10px] font-mono tabular-nums"
              >
                {formatX(xVal)}
              </text>
            </g>
          );
        })}

        {showDiagonal && (
          <line
            x1={toPxX(Math.max(minX, minY))}
            y1={toPxY(Math.max(minX, minY))}
            x2={toPxX(Math.min(maxX, maxY))}
            y2={toPxY(Math.min(maxX, maxY))}
            stroke="#0F172A"
            strokeWidth={1.5}
            strokeDasharray="6 4"
            opacity={0.6}
          />
        )}

        {points.map((pt, idx) => {
          const cx = toPxX(pt.x);
          const cy = toPxY(pt.y);
          const isSelected = activePoint === pt;
          return (
            <circle
              key={`${pt.label}-${idx}`}
              cx={cx}
              cy={cy}
              r={isSelected ? 6 : 4}
              fill={pt.group === 'Automatic' || pt.group === 'Diesel' ? '#1E40AF' : '#0EA5E9'}
              stroke="#FFFFFF"
              strokeWidth={1}
              opacity={0.82}
              className="cursor-pointer transition-transform"
              onMouseEnter={() => setActivePoint(pt)}
              onMouseLeave={() => setActivePoint(null)}
            />
          );
        })}

        <text
          x={padL + plotW / 2}
          y={height - 4}
          textAnchor="middle"
          className="fill-slate-500 text-[11px]"
        >
          {xLabel}
        </text>
        <text
          x={14}
          y={padT + plotH / 2}
          textAnchor="middle"
          transform={`rotate(-90 14 ${padT + plotH / 2})`}
          className="fill-slate-500 text-[11px]"
        >
          {yLabel}
        </text>
      </svg>

      <div className="mt-2 min-h-[30px] text-xs bg-slate-50 border border-slate-200 rounded-lg px-3 py-1.5 flex items-center justify-between text-slate-600 tabular-nums">
        {activePoint ? (
          <>
            <span className="font-medium text-slate-900 truncate max-w-[45%]">
              {activePoint.label} ({activePoint.group})
            </span>
            <span className="font-mono text-slate-800">
              {xLabel}: {formatX(activePoint.x)} · {yLabel}: {formatY(activePoint.y)}
            </span>
          </>
        ) : (
          <span>Hover over any data point to inspect exact vehicle values.</span>
        )}
      </div>
    </div>
  );
}

export function ActualVsPredictedChart({ points }: { points: ActualVsPredictedPoint[] }) {
  const formatCompact = (v: number) =>
    Math.abs(v) >= 10000000 ? `${(v / 10000000).toFixed(1)}Cr` : `${(v / 100000).toFixed(1)}L`;

  const displayPoints = useMemo(() => {
    if (points.length <= 450) return points;
    const step = Math.max(1, Math.floor(points.length / 450));
    return points.filter((_, idx) => idx % step === 0);
  }, [points]);

  return (
    <ScatterPlotChart
      title="Actual vs Predicted Car Prices (Untouched Test Split)"
      subtitle={`Showing ${displayPoints.length} sampled of ${points.length} untouched test vehicles; closer alignment to dashed diagonal indicates stronger generalization`}
      points={displayPoints.map((p) => ({
        x: p.actual,
        y: p.predicted,
        label: p.label,
        group: p.fuel,
      }))}
      xLabel="Actual Price"
      yLabel="Predicted Price"
      formatX={formatCompact}
      formatY={formatCompact}
      showDiagonal
    />
  );
}

export function FeatureImportanceChart({ items }: { items: FeatureImportanceItem[] }) {
  const maxImp = Math.max(0.01, ...items.map((i) => i.importance));
  return (
    <div className="bg-white border border-slate-200 rounded-xl p-5">
      <div className="mb-4">
        <h3 className="text-base font-semibold text-slate-900">
          CART Feature Importance (Normalized Variance Reduction)
        </h3>
        <p className="text-xs text-slate-500 mt-0.5">
          Proportion of total squared-error impurity reduction contributed by each specification across all tree splits
        </p>
      </div>
      <div className="space-y-3">
        {items.map((item) => {
          const pct = (item.importance / maxImp) * 100;
          const share = (item.importance * 100).toFixed(1);
          return (
            <div key={item.rawFeature} className="space-y-1">
              <div className="flex items-center justify-between text-sm">
                <span className="font-medium text-slate-800">{item.feature}</span>
                <span className="font-mono text-xs tabular-nums text-slate-700">{share}%</span>
              </div>
              <div className="h-2.5 bg-slate-100 rounded-full overflow-hidden">
                <div
                  className="h-full bg-blue-800 rounded-full"
                  style={{ width: `${Math.max(item.importance > 0 ? 1.5 : 0, pct)}%` }}
                />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

interface PositionedTreeNode {
  node: CartTreeNode;
  x: number;
  y: number;
  parentX?: number;
  parentY?: number;
  edgeLabel?: string;
}

export function CartTreeDiagram({
  tree,
  maxDisplayDepth = 3,
}: {
  tree: CartTreeNode;
  maxDisplayDepth?: number;
}) {
  const [depthLimit, setDepthLimit] = useState<number>(maxDisplayDepth);
  const [selectedNode, setSelectedNode] = useState<CartTreeNode>(tree);

  const { positioned, svgW, svgH } = useMemo(() => {
    const list: PositionedTreeNode[] = [];
    const levelH = 110;
    const nodeW = 152;
    let leafCounter = 0;

    const assign = (
      node: CartTreeNode,
      depth: number,
      parent?: { x: number; y: number; edgeLabel: string }
    ): number => {
      const treatAsLeaf = node.isLeaf || depth >= depthLimit || !node.left || !node.right;
      const y = 36 + depth * levelH;

      if (treatAsLeaf) {
        const x = 90 + leafCounter * (nodeW + 24);
        leafCounter++;
        list.push({
          node,
          x,
          y,
          parentX: parent?.x,
          parentY: parent?.y,
          edgeLabel: parent?.edgeLabel,
        });
        return x;
      }

      const leftX = assign(node.left!, depth + 1, { x: 0, y, edgeLabel: 'Yes (<=)' });
      const rightX = assign(node.right!, depth + 1, { x: 0, y, edgeLabel: 'No (>)' });
      const x = (leftX + rightX) / 2;

      // Update immediate children parentX
      for (const item of list) {
        if (item.node === node.left) {
          item.parentX = x;
          item.parentY = y;
          item.edgeLabel = 'Yes';
        } else if (item.node === node.right) {
          item.parentX = x;
          item.parentY = y;
          item.edgeLabel = 'No';
        }
      }

      list.push({
        node,
        x,
        y,
        parentX: parent?.x,
        parentY: parent?.y,
        edgeLabel: parent?.edgeLabel,
      });

      return x;
    };

    assign(tree, 0);
    const maxX = Math.max(760, ...list.map((p) => p.x + 110));
    const maxY = Math.max(240, ...list.map((p) => p.y + 85));
    return { positioned: list, svgW: maxX, svgH: maxY };
  }, [tree, depthLimit]);

  return (
    <div className="bg-white border border-slate-200 rounded-xl p-5 space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h3 className="text-base font-semibold text-slate-900">
            Interactive CART Decision Tree Structure
          </h3>
          <p className="text-xs text-slate-500 mt-0.5">
            Click any node to inspect split criteria, sample count, squared error, and predicted mean price
          </p>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-xs text-slate-600">Visible Depth:</span>
          {[2, 3, 4, 5].map((d) => (
            <button
              key={d}
              onClick={() => setDepthLimit(d)}
              className={`px-2.5 py-1 text-xs font-mono rounded-md border transition-colors whitespace-nowrap shrink-0 ${
                depthLimit === d
                  ? 'bg-blue-800 text-white border-blue-800'
                  : 'bg-white text-slate-700 border-slate-200 hover:bg-slate-50'
              }`}
            >
              Depth {d}
            </button>
          ))}
        </div>
      </div>

      <div className="overflow-x-auto border border-slate-200 rounded-lg bg-slate-50 p-3">
        <svg width={svgW} height={svgH} className="block mx-auto">
          {positioned.map((item) => {
            if (item.parentX === undefined || item.parentY === undefined) return null;
            const midY = (item.parentY + 46 + item.y) / 2;
            return (
              <g key={`edge-${item.node.nodeId}`}>
                <line
                  x1={item.parentX}
                  y1={item.parentY + 48}
                  x2={item.x}
                  y2={item.y}
                  stroke="#94A3B8"
                  strokeWidth={1.5}
                />
                <text
                  x={(item.parentX + item.x) / 2}
                  y={midY}
                  textAnchor="middle"
                  className="fill-slate-600 text-[10px] font-mono bg-white"
                >
                  {item.edgeLabel}
                </text>
              </g>
            );
          })}

          {positioned.map((item) => {
            const n = item.node;
            const isLeafRendered = n.isLeaf || n.depth >= depthLimit;
            const isSelected = selectedNode.nodeId === n.nodeId;
            const boxW = 144;
            const boxH = 54;
            const ruleShort =
              !n.isLeaf && n.featureName && n.threshold !== null
                ? n.featureName.includes('=')
                  ? `${n.featureName.replace('=', ' ≠ ')}`
                  : `${n.featureName} <= ${
                      Math.abs(n.threshold) >= 1000
                        ? Math.round(n.threshold).toLocaleString('en-IN')
                        : n.threshold.toFixed(1)
                    }`
                : 'Leaf Prediction';

            return (
              <g
                key={`node-${n.nodeId}`}
                transform={`translate(${item.x - boxW / 2}, ${item.y})`}
                onClick={() => setSelectedNode(n)}
                className="cursor-pointer"
              >
                <rect
                  width={boxW}
                  height={boxH}
                  rx={8}
                  fill={isLeafRendered ? '#EFF6FF' : '#FFFFFF'}
                  stroke={isSelected ? '#1E40AF' : isLeafRendered ? '#93C5FD' : '#CBD5E1'}
                  strokeWidth={isSelected ? 2 : 1}
                />
                <text
                  x={boxW / 2}
                  y={17}
                  textAnchor="middle"
                  className="fill-slate-900 text-[10px] font-mono font-semibold"
                >
                  {ruleShort.length > 21 ? `${ruleShort.slice(0, 19)}…` : ruleShort}
                </text>
                <text
                  x={boxW / 2}
                  y={32}
                  textAnchor="middle"
                  className="fill-blue-900 text-[11px] font-mono font-semibold tabular-nums"
                >
                  {formatLakh(n.prediction)}
                </text>
                <text
                  x={boxW / 2}
                  y={45}
                  textAnchor="middle"
                  className="fill-slate-500 text-[9px] font-mono tabular-nums"
                >
                  n={n.samples} · depth={n.depth}
                </text>
              </g>
            );
          })}
        </svg>
      </div>

      <div className="bg-slate-50 border border-slate-200 rounded-lg p-4 grid grid-cols-2 md:grid-cols-4 gap-4 text-xs tabular-nums">
        <div>
          <div className="text-slate-500">Selected Node</div>
          <div className="font-mono font-semibold text-slate-900 mt-0.5">
            Node #{selectedNode.nodeId} ({selectedNode.isLeaf ? 'Leaf' : 'Internal Split'})
          </div>
        </div>
        <div>
          <div className="text-slate-500">Split Criterion</div>
          <div className="font-mono font-semibold text-slate-900 mt-0.5 truncate">
            {!selectedNode.isLeaf && selectedNode.featureName && selectedNode.threshold !== null
              ? `${selectedNode.featureName} <= ${selectedNode.threshold.toFixed(2)}`
              : 'Terminal Leaf Node'}
          </div>
        </div>
        <div>
          <div className="text-slate-500">Samples & Mean Value</div>
          <div className="font-mono font-semibold text-slate-900 mt-0.5">
            {selectedNode.samples} cars · {formatInr(selectedNode.prediction)}
          </div>
        </div>
        <div>
          <div className="text-slate-500">Node MSE (Squared Error)</div>
          <div className="font-mono font-semibold text-slate-900 mt-0.5">
            {Math.round(selectedNode.squaredError).toLocaleString('en-IN')}
          </div>
        </div>
      </div>
    </div>
  );
}
