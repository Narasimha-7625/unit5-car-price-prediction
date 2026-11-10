export type DatasetSourceType = 'sample_demo' | 'uploaded_csv';

export interface ColumnProfile {
  name: string;
  mappedRole: string;
  inferredType: 'numerical' | 'categorical' | 'target';
  missingCount: number;
  uniqueCount: number;
  sampleValues: string[];
  medianOrMode: string;
}

export interface PreprocessingReport {
  rawRows: number;
  rawColumns: number;
  cleanedRows: number;
  duplicatesRemoved: number;
  invalidRowsRemoved: number;
  invalidReasons: { reason: string; count: number }[];
  missingValuesTotal: number;
  imputedCount: number;
  columnProfiles: ColumnProfile[];
  stepsApplied: string[];
}

export interface FeatureFieldSchema {
  key: string;
  label: string;
  kind: 'numerical' | 'categorical';
  unit?: string;
  min?: number;
  max?: number;
  step?: number;
  options?: string[];
  defaultValue: string | number;
  brandModelsMap?: Record<string, string[]>;
}

export interface PreprocessorState {
  activeNumericFeatures: string[];
  activeCategoricalFeatures: string[];
  numericMedians: Record<string, number>;
  categoricalModes: Record<string, string>;
  categoricalVocab: Record<string, string[]>;
  featureNames: string[];
  featureSchema: FeatureFieldSchema[];
  hasBrandModelHierarchy: boolean;
}

export interface CartHyperparameters {
  max_depth: number;
  min_samples_split: number;
  min_samples_leaf: number;
  ccp_alpha: number;
}

export interface GridSearchCandidate {
  params: CartHyperparameters;
  meanCvR2: number;
  meanCvRmse: number;
  stdCvR2: number;
}

export interface CartTreeNode {
  nodeId: number;
  depth: number;
  isLeaf: boolean;
  samples: number;
  squaredError: number;
  prediction: number;
  featureIndex: number | null;
  featureName: string | null;
  threshold: number | null;
  impurityDecrease: number;
  left: CartTreeNode | null;
  right: CartTreeNode | null;
}

export interface DecisionPathStep {
  nodeId: number;
  depth: number;
  featureName: string;
  threshold: number;
  observedValue: number;
  direction: 'left' | 'right';
  ruleText: string;
  samples: number;
  nodeMeanPrice: number;
}

export interface EvaluationMetrics {
  mae: number;
  mse: number;
  rmse: number;
  r2: number;
  trainR2: number;
  trainRmse: number;
  trainCount: number;
  testCount: number;
}

export interface FeatureImportanceItem {
  feature: string;
  rawFeature: string;
  importance: number;
}

export interface ActualVsPredictedPoint {
  index: number;
  label: string;
  actual: number;
  predicted: number;
  residual: number;
  year: number;
  fuel: string;
}

export interface DatasetExplorerCharts {
  priceHistogram: { rangeLabel: string; min: number; max: number; count: number }[];
  priceByFuel: { fuel: string; count: number; avgPrice: number; minPrice: number; maxPrice: number; medianPrice: number }[];
  priceByTransmission: { transmission: string; count: number; avgPrice: number; minPrice: number; maxPrice: number; medianPrice: number }[];
  yearVsPrice: { year: number; price: number; label: string; fuel: string }[];
  kmVsPrice: { km: number; price: number; label: string; transmission: string }[];
  residualHistogram: { rangeLabel: string; min: number; max: number; count: number }[];
}

export interface CleanedCarRow {
  brand?: string;
  model?: string;
  year?: number;
  selling_price: number;
  km_driven?: number;
  fuel?: string;
  transmission?: string;
  owner?: string;
  mileage?: number;
  engine?: number;
  max_power?: number;
  seats?: number;
  [key: string]: string | number | undefined;
}

export interface CartModelArtifact {
  modelVersion: string;
  trainedAt: string;
  datasetName: string;
  datasetSource: DatasetSourceType;
  algorithm: 'CART DecisionTreeRegressor';
  criterion: 'squared_error';
  randomState: number;
  trainTestRatio: string;
  preprocessing: PreprocessingReport;
  preprocessorState: PreprocessorState;
  bestParams: CartHyperparameters;
  gridSearchSummary: {
    totalCombinationsTested: number;
    cvFolds: number;
    topCandidates: GridSearchCandidate[];
  };
  metrics: EvaluationMetrics;
  featureImportances: FeatureImportanceItem[];
  tree: CartTreeNode;
  treeStats: {
    totalNodes: number;
    leafNodes: number;
    actualMaxDepth: number;
  };
  actualVsPredicted: ActualVsPredictedPoint[];
  charts: DatasetExplorerCharts;
  previewRows: CleanedCarRow[];
  totalCleanedRows: number;
}

export interface PredictionRequest {
  [key: string]: string | number | undefined;
}

export interface PredictionResponse {
  predictedPriceInr: number;
  formattedInr: string;
  formattedLakh: string;
  validatedInputs: Record<string, string | number>;
  decisionPath: DecisionPathStep[];
  leafSamples: number;
  leafSquaredError: number;
  modelTrainedAt: string;
  datasetSource: DatasetSourceType;
}
