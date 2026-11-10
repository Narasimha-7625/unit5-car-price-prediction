import {
  CartDecisionTreeRegressor,
  computeTreeStats,
  predictTreeRow,
} from './cartRegressor.ts';
import {
  cleanAndInspectDataset,
  fitPreprocessorOnTrain,
  formatInr,
  formatLakh,
  transformRows,
  validatePredictionInputs,
} from './preprocessing.ts';
import type {
  ActualVsPredictedPoint,
  CartHyperparameters,
  CartModelArtifact,
  CleanedCarRow,
  DatasetExplorerCharts,
  DatasetSourceType,
  EvaluationMetrics,
  FeatureImportanceItem,
  GridSearchCandidate,
  PredictionRequest,
  PredictionResponse,
} from './types.ts';

/**
 * Deterministic seeded PRNG (Mulberry32) for reproducible 80:20 train-test split and CV folds.
 */
function createSeededRng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function seededShuffle<T>(arr: T[], seed: number): T[] {
  const copy = [...arr];
  const rand = createSeededRng(seed);
  for (let i = copy.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [copy[i], copy[j]] = [copy[j], copy[i]];
  }
  return copy;
}

function computeRegressionMetrics(yTrue: number[], yPred: number[]): {
  mae: number;
  mse: number;
  rmse: number;
  r2: number;
} {
  const n = yTrue.length;
  if (n === 0) return { mae: 0, mse: 0, rmse: 0, r2: 0 };

  let sumAbs = 0;
  let sumSq = 0;
  let sumY = 0;
  for (let i = 0; i < n; i++) {
    sumY += yTrue[i];
  }
  const meanY = sumY / n;

  let ssTot = 0;
  for (let i = 0; i < n; i++) {
    const err = yTrue[i] - yPred[i];
    sumAbs += Math.abs(err);
    sumSq += err * err;
    const dev = yTrue[i] - meanY;
    ssTot += dev * dev;
  }

  const mae = sumAbs / n;
  const mse = sumSq / n;
  const rmse = Math.sqrt(mse);
  const r2 = ssTot > 1e-9 ? 1 - sumSq / ssTot : 0;
  return { mae, mse, rmse, r2 };
}

function median(values: number[]): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)];
}

function buildDatasetCharts(
  cleanedRows: CleanedCarRow[],
  residuals: number[]
): DatasetExplorerCharts {
  // 1. Price Distribution Histogram
  const formatShortRangeVal = (v: number): string => {
    const abs = Math.abs(v);
    if (abs >= 10000000) return `${(v / 10000000).toFixed(1)}Cr`;
    const lakhs = v / 100000;
    return Math.abs(lakhs) >= 10 ? `${Math.round(lakhs)}L` : `${lakhs.toFixed(1)}L`;
  };

  const prices = cleanedRows.map((r) => r.selling_price);
  const minP = Math.min(...prices);
  const maxP = Math.max(...prices);
  const binCount = 8;
  const binWidth = Math.max(100000, (maxP - minP) / binCount);
  const priceHistogram = Array.from({ length: binCount }, (_, b) => {
    const start = minP + b * binWidth;
    const end = b === binCount - 1 ? maxP + 1 : start + binWidth;
    const count = prices.filter((p) => p >= start && p < end).length;
    return {
      rangeLabel: `${formatShortRangeVal(start)}–${formatShortRangeVal(end)}`,
      min: start,
      max: end,
      count,
    };
  });

  // 2. Price by Fuel Type
  const fuelMap = new Map<string, number[]>();
  for (const r of cleanedRows) {
    const key = r.fuel ?? 'Unspecified';
    if (!fuelMap.has(key)) fuelMap.set(key, []);
    fuelMap.get(key)!.push(r.selling_price);
  }
  const priceByFuel = Array.from(fuelMap.entries())
    .map(([fuel, arr]) => ({
      fuel,
      count: arr.length,
      avgPrice: arr.reduce((a, b) => a + b, 0) / arr.length,
      minPrice: Math.min(...arr),
      maxPrice: Math.max(...arr),
      medianPrice: median(arr),
    }))
    .sort((a, b) => b.count - a.count);

  // 3. Price by Transmission
  const transMap = new Map<string, number[]>();
  for (const r of cleanedRows) {
    const key = r.transmission ?? 'Unspecified';
    if (!transMap.has(key)) transMap.set(key, []);
    transMap.get(key)!.push(r.selling_price);
  }
  const priceByTransmission = Array.from(transMap.entries())
    .map(([transmission, arr]) => ({
      transmission,
      count: arr.length,
      avgPrice: arr.reduce((a, b) => a + b, 0) / arr.length,
      minPrice: Math.min(...arr),
      maxPrice: Math.max(...arr),
      medianPrice: median(arr),
    }))
    .sort((a, b) => b.count - a.count);

  // Sample up to 350 scatter points evenly for responsive SVG rendering
  const step = Math.max(1, Math.floor(cleanedRows.length / 350));
  const sampled = cleanedRows.filter((_, idx) => idx % step === 0);

  const yearVsPrice = sampled
    .filter((r) => typeof r.year === 'number')
    .map((r) => ({
      year: r.year as number,
      price: r.selling_price,
      label: [r.brand, r.model].filter(Boolean).join(' ') || 'Vehicle',
      fuel: r.fuel ?? 'Unknown',
    }));

  const kmVsPrice = sampled
    .filter((r) => typeof r.km_driven === 'number')
    .map((r) => ({
      km: r.km_driven as number,
      price: r.selling_price,
      label: [r.brand, r.model].filter(Boolean).join(' ') || 'Vehicle',
      transmission: r.transmission ?? 'Unknown',
    }));

  // Residual Distribution Histogram
  const minR = residuals.length > 0 ? Math.min(...residuals) : -100000;
  const maxR = residuals.length > 0 ? Math.max(...residuals) : 100000;
  const rBins = 9;
  const rSpan = Math.max(50000, maxR - minR);
  const rWidth = rSpan / rBins;
  const residualHistogram = Array.from({ length: rBins }, (_, b) => {
    const start = minR + b * rWidth;
    const end = b === rBins - 1 ? maxR + 1 : start + rWidth;
    const count = residuals.filter((v) => v >= start && v < end).length;
    return {
      rangeLabel: formatShortRangeVal(start),
      min: start,
      max: end,
      count,
    };
  });

  return {
    priceHistogram,
    priceByFuel,
    priceByTransmission,
    yearVsPrice,
    kmVsPrice,
    residualHistogram,
  };
}

/**
 * Full end-to-end CART training pipeline:
 * 1. Clean & inspect CSV
 * 2. Reproducible 80:20 split BEFORE fitting preprocessor (prevents leakage)
 * 3. 5-Fold GridSearchCV on training split only across CART hyperparameters
 * 4. Retrain best CART DecisionTreeRegressor on full 80% training split
 * 5. Evaluate on untouched 20% test split
 */
export function trainCartPipelineFromCsv(
  csvText: string,
  datasetName: string,
  datasetSource: DatasetSourceType
): CartModelArtifact {
  const randomState = 42;
  const {
    cleanedRows,
    report,
    activeNumericFeatures,
    activeCategoricalFeatures,
  } = cleanAndInspectDataset(csvText);

  const shuffled = seededShuffle(cleanedRows, randomState);
  const splitIdx = Math.max(15, Math.floor(shuffled.length * 0.8));
  const trainRows = shuffled.slice(0, splitIdx);
  const testRows = shuffled.slice(splitIdx);

  // Fit preprocessor strictly on training split
  const preprocessorState = fitPreprocessorOnTrain(
    trainRows,
    cleanedRows,
    activeNumericFeatures,
    activeCategoricalFeatures
  );

  const XTrain = transformRows(trainRows, preprocessorState);
  const yTrain = trainRows.map((r) => r.selling_price);
  const XTest = transformRows(testRows, preprocessorState);
  const yTest = testRows.map((r) => r.selling_price);

  // Compute target variance scale for reasonable ccp_alpha grid
  const meanTrainY = yTrain.reduce((a, b) => a + b, 0) / yTrain.length;
  const trainVar =
    yTrain.reduce((acc, v) => acc + (v - meanTrainY) * (v - meanTrainY), 0) / yTrain.length;

  const maxDepthGrid = [4, 6, 8, 10, 14];
  const minSamplesSplitGrid = [2, 5, 10];
  const minSamplesLeafGrid = [1, 3, 5];
  const ccpAlphaGrid = [0, Math.round(trainVar * 0.002), Math.round(trainVar * 0.01)];

  const paramGrid: CartHyperparameters[] = [];
  for (const max_depth of maxDepthGrid) {
    for (const min_samples_split of minSamplesSplitGrid) {
      for (const min_samples_leaf of minSamplesLeafGrid) {
        for (const ccp_alpha of ccpAlphaGrid) {
          paramGrid.push({ max_depth, min_samples_split, min_samples_leaf, ccp_alpha });
        }
      }
    }
  }

  // 5-Fold Cross-Validation strictly on training data
  const cvFolds = Math.min(5, Math.max(3, Math.floor(trainRows.length / 8)));
  const foldAssignments = trainRows.map((_, idx) => idx % cvFolds);
  const candidates: GridSearchCandidate[] = [];

  for (const params of paramGrid) {
    const foldR2s: number[] = [];
    const foldRmses: number[] = [];

    for (let fold = 0; fold < cvFolds; fold++) {
      const foldTrainX: number[][] = [];
      const foldTrainY: number[] = [];
      const foldValX: number[][] = [];
      const foldValY: number[] = [];

      for (let i = 0; i < XTrain.length; i++) {
        if (foldAssignments[i] === fold) {
          foldValX.push(XTrain[i]);
          foldValY.push(yTrain[i]);
        } else {
          foldTrainX.push(XTrain[i]);
          foldTrainY.push(yTrain[i]);
        }
      }

      const cvModel = new CartDecisionTreeRegressor(params, preprocessorState.featureNames);
      const cvTree = cvModel.fit(foldTrainX, foldTrainY);
      const valPreds = foldValX.map((row) => predictTreeRow(cvTree, row).prediction);
      const m = computeRegressionMetrics(foldValY, valPreds);
      foldR2s.push(m.r2);
      foldRmses.push(m.rmse);
    }

    const meanCvR2 = foldR2s.reduce((a, b) => a + b, 0) / cvFolds;
    const meanCvRmse = foldRmses.reduce((a, b) => a + b, 0) / cvFolds;
    const varianceR2 =
      foldR2s.reduce((acc, v) => acc + (v - meanCvR2) * (v - meanCvR2), 0) / cvFolds;

    candidates.push({
      params,
      meanCvR2,
      meanCvRmse,
      stdCvR2: Math.sqrt(varianceR2),
    });
  }

  candidates.sort((a, b) => b.meanCvR2 - a.meanCvR2);
  const bestParams = candidates[0].params;

  // Train final CART model on full 80% training split
  const finalModel = new CartDecisionTreeRegressor(bestParams, preprocessorState.featureNames);
  const finalTree = finalModel.fit(XTrain, yTrain);

  // Evaluate on training split and untouched 20% test split
  const trainPreds = XTrain.map((row) => predictTreeRow(finalTree, row).prediction);
  const trainMetrics = computeRegressionMetrics(yTrain, trainPreds);

  const testPreds = XTest.map((row) => predictTreeRow(finalTree, row).prediction);
  const testMetrics = computeRegressionMetrics(yTest, testPreds);

  const metrics: EvaluationMetrics = {
    mae: testMetrics.mae,
    mse: testMetrics.mse,
    rmse: testMetrics.rmse,
    r2: testMetrics.r2,
    trainR2: trainMetrics.r2,
    trainRmse: trainMetrics.rmse,
    trainCount: trainRows.length,
    testCount: testRows.length,
  };

  // Aggregate feature importances back to raw specification columns as well as encoded features
  const rawImportances = finalModel.computeFeatureImportances();
  const groupedImportance = new Map<string, number>();
  for (let i = 0; i < preprocessorState.featureNames.length; i++) {
    const fname = preprocessorState.featureNames[i];
    const baseName = fname.includes('=') ? fname.split('=')[0] : fname;
    groupedImportance.set(baseName, (groupedImportance.get(baseName) ?? 0) + rawImportances[i]);
  }

  const featureImportances: FeatureImportanceItem[] = Array.from(groupedImportance.entries())
    .map(([rawFeature, importance]) => ({
      feature: formatFeatureDisplayName(rawFeature),
      rawFeature,
      importance,
    }))
    .sort((a, b) => b.importance - a.importance);

  const actualVsPredicted: ActualVsPredictedPoint[] = testRows.map((row, idx) => {
    const actual = yTest[idx];
    const predicted = testPreds[idx];
    return {
      index: idx + 1,
      label: [row.brand, row.model].filter(Boolean).join(' ') || `Car #${idx + 1}`,
      actual,
      predicted,
      residual: actual - predicted,
      year: row.year ?? 0,
      fuel: row.fuel ?? 'Unknown',
    };
  });

  const residuals = actualVsPredicted.map((p) => p.residual);
  const charts = buildDatasetCharts(cleanedRows, residuals);
  const treeStats = computeTreeStats(finalTree);

  return {
    modelVersion: '1.0.0',
    trainedAt: new Date().toISOString(),
    datasetName,
    datasetSource,
    algorithm: 'CART DecisionTreeRegressor',
    criterion: 'squared_error',
    randomState,
    trainTestRatio: '80:20',
    preprocessing: report,
    preprocessorState,
    bestParams,
    gridSearchSummary: {
      totalCombinationsTested: paramGrid.length,
      cvFolds,
      topCandidates: candidates.slice(0, 6),
    },
    metrics,
    featureImportances,
    tree: finalTree,
    treeStats,
    actualVsPredicted,
    charts,
    previewRows: cleanedRows.slice(0, 100),
    totalCleanedRows: cleanedRows.length,
  };
}

export function predictFromArtifact(
  artifact: CartModelArtifact,
  request: PredictionRequest
): { ok: true; result: PredictionResponse } | { ok: false; errors: Record<string, string> } {
  const { validated, errors } = validatePredictionInputs(request, artifact.preprocessorState);
  if (Object.keys(errors).length > 0) {
    return { ok: false, errors };
  }

  const [featureVector] = transformRows([validated], artifact.preprocessorState);
  const { prediction, path, leafSamples, leafSquaredError } = predictTreeRow(
    artifact.tree,
    featureVector
  );
  const clampedPrice = Math.max(25000, Math.round(prediction));

  return {
    ok: true,
    result: {
      predictedPriceInr: clampedPrice,
      formattedInr: formatInr(clampedPrice),
      formattedLakh: formatLakh(clampedPrice),
      validatedInputs: validated,
      decisionPath: path,
      leafSamples,
      leafSquaredError,
      modelTrainedAt: artifact.trainedAt,
      datasetSource: artifact.datasetSource,
    },
  };
}

function formatFeatureDisplayName(key: string): string {
  const map: Record<string, string> = {
    year: 'Manufacturing Year',
    km_driven: 'Kilometres Driven',
    engine: 'Engine Capacity (CC)',
    mileage: 'Mileage (kmpl)',
    max_power: 'Maximum Power (bhp)',
    seats: 'Number of Seats',
    brand: 'Car Brand',
    model: 'Car Model',
    fuel: 'Fuel Type',
    transmission: 'Transmission',
    owner: 'Ownership History',
  };
  return map[key] ?? key;
}
