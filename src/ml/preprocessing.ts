import type {
  CleanedCarRow,
  ColumnProfile,
  FeatureFieldSchema,
  PreprocessingReport,
  PreprocessorState,
  PredictionRequest,
} from './types.ts';

const NUMERIC_FEATURE_META: Record<
  string,
  { label: string; unit: string; min: number; max: number; step: number }
> = {
  year: { label: 'Manufacturing Year', unit: 'year', min: 1990, max: new Date().getFullYear(), step: 1 },
  km_driven: { label: 'Kilometres Driven', unit: 'km', min: 0, max: 600000, step: 500 },
  engine: { label: 'Engine Capacity', unit: 'CC', min: 0, max: 6000, step: 10 },
  mileage: { label: 'Mileage', unit: 'kmpl', min: 0, max: 50, step: 0.1 },
  max_power: { label: 'Maximum Power', unit: 'bhp', min: 20, max: 600, step: 0.5 },
  seats: { label: 'Number of Seats', unit: 'seats', min: 2, max: 10, step: 1 },
};

const CATEGORICAL_FEATURE_META: Record<string, { label: string }> = {
  brand: { label: 'Car Brand' },
  model: { label: 'Car Model' },
  fuel: { label: 'Fuel Type' },
  transmission: { label: 'Transmission Type' },
  owner: { label: 'Ownership History' },
};

const MULTI_WORD_BRANDS = [
  'Maruti Suzuki',
  'Land Rover',
  'Aston Martin',
  'Rolls Royce',
  'Mercedes Benz',
  'Mercedes-Benz',
  'Force Motors',
  'MG Motor',
];

export function formatInr(amount: number): string {
  const rounded = Math.max(0, Math.round(amount));
  return `₹${rounded.toLocaleString('en-IN')}`;
}

export function formatLakh(amount: number): string {
  const safe = Math.max(0, amount);
  if (safe >= 10000000) {
    return `₹${(safe / 10000000).toFixed(2)} crore`;
  }
  return `₹${(safe / 100000).toFixed(2)} lakh`;
}

/**
 * RFC-4180 compliant CSV parser supporting quoted commas and CRLF/LF newlines.
 */
export function parseCsvText(csvText: string): { headers: string[]; rows: Record<string, string>[] } {
  const cleanedText = csvText.replace(/^\uFEFF/, '').trim();
  if (!cleanedText) {
    throw new Error('The uploaded CSV file is empty.');
  }

  const matrix: string[][] = [];
  let currentRow: string[] = [];
  let currentCell = '';
  let inQuotes = false;

  for (let i = 0; i < cleanedText.length; i++) {
    const ch = cleanedText[i];
    const next = cleanedText[i + 1];

    if (inQuotes) {
      if (ch === '"' && next === '"') {
        currentCell += '"';
        i++;
      } else if (ch === '"') {
        inQuotes = false;
      } else {
        currentCell += ch;
      }
    } else {
      if (ch === '"') {
        inQuotes = true;
      } else if (ch === ',') {
        currentRow.push(currentCell.trim());
        currentCell = '';
      } else if (ch === '\r' && next === '\n') {
        currentRow.push(currentCell.trim());
        matrix.push(currentRow);
        currentRow = [];
        currentCell = '';
        i++;
      } else if (ch === '\n' || ch === '\r') {
        currentRow.push(currentCell.trim());
        matrix.push(currentRow);
        currentRow = [];
        currentCell = '';
      } else {
        currentCell += ch;
      }
    }
  }

  if (currentCell.length > 0 || currentRow.length > 0) {
    currentRow.push(currentCell.trim());
    matrix.push(currentRow);
  }

  if (matrix.length < 2) {
    throw new Error('CSV must include a header row and at least one data record.');
  }

  const headers = matrix[0].map((h, idx) => {
    const trimmed = h.trim();
    return trimmed ? trimmed : `Unnamed: ${idx}`;
  });
  if (headers.length < 3) {
    throw new Error('CSV format invalid: expected at least 3 columns including price and vehicle specifications.');
  }

  const rows: Record<string, string>[] = [];
  for (let r = 1; r < matrix.length; r++) {
    const line = matrix[r];
    if (line.length === 1 && line[0] === '') continue;
    const obj: Record<string, string> = {};
    for (let c = 0; c < headers.length; c++) {
      obj[headers[c]] = (line[c] ?? '').trim();
    }
    rows.push(obj);
  }

  return { headers, rows };
}

function normalizeHeader(h: string): string {
  return h.toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_|_$/g, '');
}

function detectColumnMapping(headers: string[]): Record<string, string> {
  const mapping: Record<string, string> = {};

  for (const h of headers) {
    const norm = normalizeHeader(h);
    if (!norm || norm.startsWith('unnamed')) continue;

    if (
      [
        'selling_price',
        'price',
        'resale_price',
        'used_price',
        'target_price',
        'sellingprice',
        'car_prices_in_rupee',
        'car_price_in_rupee',
        'car_price',
      ].includes(norm)
    ) {
      mapping.selling_price = mapping.selling_price ?? h;
    } else if (['brand', 'make', 'company', 'manufacturer'].includes(norm)) {
      mapping.brand = mapping.brand ?? h;
    } else if (['model', 'car_model', 'variant'].includes(norm)) {
      mapping.model = mapping.model ?? h;
    } else if (['name', 'car_name', 'vehicle_name', 'title'].includes(norm)) {
      mapping.name = mapping.name ?? h;
    } else if (
      ['year', 'model_year', 'manufacturing_year', 'reg_year', 'manufacture', 'manufactured_year'].includes(
        norm
      )
    ) {
      mapping.year = mapping.year ?? h;
    } else if (
      ['km_driven', 'kms_driven', 'kilometers_driven', 'kilometres_driven', 'mileage_km', 'odometer'].includes(
        norm
      )
    ) {
      mapping.km_driven = mapping.km_driven ?? h;
    } else if (['fuel', 'fuel_type'].includes(norm)) {
      mapping.fuel = mapping.fuel ?? h;
    } else if (['transmission', 'transmission_type', 'gearbox'].includes(norm)) {
      mapping.transmission = mapping.transmission ?? h;
    } else if (['owner', 'owner_type', 'ownership', 'owners'].includes(norm)) {
      mapping.owner = mapping.owner ?? h;
    } else if (['mileage', 'fuel_economy'].includes(norm)) {
      mapping.mileage = mapping.mileage ?? h;
    } else if (['engine', 'engine_cc', 'engine_capacity', 'cc'].includes(norm)) {
      mapping.engine = mapping.engine ?? h;
    } else if (['max_power', 'power', 'bhp', 'horsepower'].includes(norm)) {
      mapping.max_power = mapping.max_power ?? h;
    } else if (['seats', 'seating_capacity', 'no_of_seats'].includes(norm)) {
      mapping.seats = mapping.seats ?? h;
    }
  }

  return mapping;
}

export function parseIndianPrice(raw: string | undefined): number | null {
  if (!raw) return null;
  const trimmed = raw.trim();
  if (!trimmed || ['na', 'nan', 'null', 'none', '-'].includes(trimmed.toLowerCase())) return null;

  const lower = trimmed.toLowerCase();
  const numericMatch = lower.replace(/,/g, '').match(/-?\d+(\.\d+)?/);
  if (!numericMatch) return null;
  const baseNum = parseFloat(numericMatch[0]);
  if (!Number.isFinite(baseNum)) return null;

  if (lower.includes('crore') || /\bcr\b/.test(lower)) {
    return baseNum * 10000000;
  }
  if (lower.includes('lakh') || lower.includes('lac')) {
    return baseNum * 100000;
  }
  if (lower.includes('thousand') || /\bk\b/.test(lower)) {
    return baseNum * 1000;
  }
  return baseNum;
}

export function parseNumericWithUnit(raw: string | undefined): number | null {
  if (!raw) return null;
  const trimmed = raw.trim();
  if (!trimmed || ['na', 'nan', 'null', 'none', '-'].includes(trimmed.toLowerCase())) return null;
  const cleaned = trimmed.replace(/,/g, '');
  const match = cleaned.match(/-?\d+(\.\d+)?/);
  if (!match) return null;
  const val = parseFloat(match[0]);
  return Number.isFinite(val) ? val : null;
}

function splitBrandAndModel(fullName: string): { brand: string; model: string } {
  const clean = fullName.trim();
  for (const multi of MULTI_WORD_BRANDS) {
    if (clean.toLowerCase().startsWith(multi.toLowerCase())) {
      const rest = clean.slice(multi.length).trim();
      return { brand: multi, model: rest || multi };
    }
  }
  const parts = clean.split(/\s+/);
  if (parts.length === 1) return { brand: parts[0], model: parts[0] };
  return { brand: parts[0], model: parts.slice(1).join(' ') };
}

function normalizeOwnerLabel(raw: string): string {
  const lower = raw.toLowerCase().trim();
  if (lower === '0' || lower.startsWith('0th')) return '0th Owner';
  if (lower.includes('first') || lower === '1' || lower.startsWith('1st')) return 'First Owner';
  if (lower.includes('second') || lower === '2' || lower.startsWith('2nd')) return 'Second Owner';
  if (lower.includes('third') || lower === '3' || lower.startsWith('3rd')) return 'Third Owner';
  if (lower.includes('fourth') || lower.includes('fifth') || lower.startsWith('4') || lower.startsWith('5')) {
    return 'Fourth & Above Owner';
  }
  if (lower.includes('test')) return 'Test Drive Car';
  return raw.trim();
}

function normalizeFuelLabel(raw: string): string {
  const lower = raw.toLowerCase().trim();
  if (lower === 'cng') return 'CNG';
  if (lower === 'lpg') return 'LPG';
  if (lower === 'petrol') return 'Petrol';
  if (lower === 'diesel') return 'Diesel';
  if (lower === 'electric' || lower === 'ev') return 'Electric';
  return raw.trim();
}

export function cleanAndInspectDataset(csvText: string): {
  cleanedRows: CleanedCarRow[];
  report: PreprocessingReport;
  activeNumericFeatures: string[];
  activeCategoricalFeatures: string[];
} {
  const { headers, rows } = parseCsvText(csvText);
  const colMap = detectColumnMapping(headers);

  if (!colMap.selling_price) {
    throw new Error(
      'Required target price column not found. Include a column named "selling_price" or "price".'
    );
  }

  const hasNameSplit = !colMap.brand && Boolean(colMap.name);
  const stepsApplied: string[] = [];
  stepsApplied.push(`Loaded CSV dataset with ${rows.length} raw records across ${headers.length} columns.`);

  // Check if bare numeric prices are stored in Lakhs (e.g., 4.5, 6.8)
  const rawPriceSamples = rows
    .map((r) => parseIndianPrice(r[colMap.selling_price]))
    .filter((v): v is number => v !== null && v > 0);

  rawPriceSamples.sort((a, b) => a - b);
  const medianRawPrice =
    rawPriceSamples.length > 0 ? rawPriceSamples[Math.floor(rawPriceSamples.length / 2)] : 0;
  const priceAssumedLakhs = medianRawPrice > 0 && medianRawPrice < 250;

  if (priceAssumedLakhs) {
    stepsApplied.push(
      'Detected bare numeric prices below 250 (median < 250); converted values from Lakhs to Indian Rupees (× 1,00,000).'
    );
  } else {
    stepsApplied.push(
      'Parsed currency symbols (₹), Indian commas, and unit suffixes (Lakh, Cr, Thousand) into numeric INR values.'
    );
  }

  if (hasNameSplit) {
    stepsApplied.push('Extracted structured "brand" and "model" features from combined vehicle "name" column.');
  }

  stepsApplied.push(
    'Converted text-formatted specifications (e.g., "21.21 kmpl", "1197 CC", "81.80 bhp") into pure floating-point numeric values.'
  );

  // Determine which specification columns actually exist in this CSV
  const candidateNumeric = ['year', 'km_driven', 'engine', 'mileage', 'max_power', 'seats'];
  const activeNumericFeatures = candidateNumeric.filter((k) => Boolean(colMap[k]));

  const activeCategoricalFeatures: string[] = [];
  if (colMap.brand || hasNameSplit) activeCategoricalFeatures.push('brand');
  if (colMap.model || hasNameSplit) activeCategoricalFeatures.push('model');
  if (colMap.fuel) activeCategoricalFeatures.push('fuel');
  if (colMap.transmission) activeCategoricalFeatures.push('transmission');
  if (colMap.owner) activeCategoricalFeatures.push('owner');

  if (activeNumericFeatures.length + activeCategoricalFeatures.length < 2) {
    throw new Error(
      'Dataset has too few valid vehicle specification features. Include columns like year, km_driven, fuel, transmission, brand, etc.'
    );
  }

  // Count missing values per mapped column in raw dataset
  const missingCounts: Record<string, number> = {};
  const uniqueValueSets: Record<string, Set<string>> = {};
  const sampleValueMap: Record<string, string[]> = {};

  for (const h of headers) {
    missingCounts[h] = 0;
    uniqueValueSets[h] = new Set();
    sampleValueMap[h] = [];
  }

  let totalMissing = 0;
  for (const row of rows) {
    for (const h of headers) {
      const v = (row[h] ?? '').trim();
      if (!v || ['na', 'nan', 'null', 'none', '-'].includes(v.toLowerCase())) {
        missingCounts[h]++;
        totalMissing++;
      } else {
        uniqueValueSets[h].add(v);
        if (sampleValueMap[h].length < 3 && !sampleValueMap[h].includes(v)) {
          sampleValueMap[h].push(v);
        }
      }
    }
  }

  // Filter invalid & duplicate records (ignoring synthetic unnamed row-index columns)
  const dedupHeaders = headers.filter((h) => !normalizeHeader(h).startsWith('unnamed'));
  const currentMaxYear = new Date().getFullYear() + 1;
  const seenSignatures = new Set<string>();
  let duplicatesRemoved = 0;
  let invalidRowsRemoved = 0;
  const invalidReasonMap: Record<string, number> = {};

  const recordInvalid = (reason: string) => {
    invalidRowsRemoved++;
    invalidReasonMap[reason] = (invalidReasonMap[reason] ?? 0) + 1;
  };

  const interimRows: CleanedCarRow[] = [];

  for (const rawRow of rows) {
    // Deduplicate exact raw records across real data columns
    const signature = dedupHeaders.map((h) => (rawRow[h] ?? '').trim().toLowerCase()).join('||');
    if (seenSignatures.has(signature)) {
      duplicatesRemoved++;
      continue;
    }
    seenSignatures.add(signature);

    let price = parseIndianPrice(rawRow[colMap.selling_price]);
    if (price !== null && priceAssumedLakhs) {
      price = price * 100000;
    }

    if (price === null || price <= 5000 || price > 500000000) {
      recordInvalid('Missing, non-positive, or unrealistic selling price');
      continue;
    }

    const parsedYear = colMap.year ? parseNumericWithUnit(rawRow[colMap.year]) : null;
    if (colMap.year && parsedYear !== null && (parsedYear < 1985 || parsedYear > currentMaxYear)) {
      recordInvalid(`Unrealistic manufacturing year (${parsedYear})`);
      continue;
    }

    const parsedKm = colMap.km_driven ? parseNumericWithUnit(rawRow[colMap.km_driven]) : null;
    if (colMap.km_driven && parsedKm !== null && (parsedKm < 0 || parsedKm > 2000000)) {
      recordInvalid(`Invalid kilometres driven (${parsedKm})`);
      continue;
    }

    const parsedEngine = colMap.engine ? parseNumericWithUnit(rawRow[colMap.engine]) : null;
    if (colMap.engine && parsedEngine !== null && (parsedEngine < 0 || parsedEngine > 8000)) {
      recordInvalid(`Invalid engine capacity (${parsedEngine} CC)`);
      continue;
    }

    const parsedMileage = colMap.mileage ? parseNumericWithUnit(rawRow[colMap.mileage]) : null;
    if (colMap.mileage && parsedMileage !== null && (parsedMileage < 0 || parsedMileage > 100)) {
      recordInvalid(`Invalid mileage value (${parsedMileage})`);
      continue;
    }

    const parsedPower = colMap.max_power ? parseNumericWithUnit(rawRow[colMap.max_power]) : null;
    if (colMap.max_power && parsedPower !== null && (parsedPower < 0 || parsedPower > 1500)) {
      recordInvalid(`Invalid max power (${parsedPower} bhp)`);
      continue;
    }

    const parsedSeats = colMap.seats ? parseNumericWithUnit(rawRow[colMap.seats]) : null;
    if (colMap.seats && parsedSeats !== null && (parsedSeats < 2 || parsedSeats > 14)) {
      recordInvalid(`Invalid seat count (${parsedSeats})`);
      continue;
    }

    const cleaned: CleanedCarRow = { selling_price: Math.round(price) };

    if (hasNameSplit && colMap.name) {
      const split = splitBrandAndModel(rawRow[colMap.name] ?? '');
      cleaned.brand = split.brand || undefined;
      cleaned.model = split.model || undefined;
    } else {
      if (colMap.brand) cleaned.brand = rawRow[colMap.brand]?.trim() || undefined;
      if (colMap.model) cleaned.model = rawRow[colMap.model]?.trim() || undefined;
    }

    if (colMap.year && parsedYear !== null) cleaned.year = Math.round(parsedYear);
    if (colMap.km_driven && parsedKm !== null) cleaned.km_driven = Math.round(parsedKm);
    if (colMap.engine && parsedEngine !== null) cleaned.engine = parsedEngine;
    if (colMap.mileage && parsedMileage !== null) cleaned.mileage = parsedMileage;
    if (colMap.max_power && parsedPower !== null) cleaned.max_power = parsedPower;
    if (colMap.seats && parsedSeats !== null) cleaned.seats = Math.round(parsedSeats);

    if (colMap.fuel && rawRow[colMap.fuel]?.trim()) {
      cleaned.fuel = normalizeFuelLabel(rawRow[colMap.fuel]);
    }
    if (colMap.transmission && rawRow[colMap.transmission]?.trim()) {
      cleaned.transmission = rawRow[colMap.transmission].trim();
    }
    if (colMap.owner && rawRow[colMap.owner]?.trim()) {
      cleaned.owner = normalizeOwnerLabel(rawRow[colMap.owner]);
    }

    interimRows.push(cleaned);
  }

  if (interimRows.length < 25) {
    throw new Error(
      `Only ${interimRows.length} valid records remained after cleaning. At least 25 valid records are required for an 80:20 train-test split and cross-validation.`
    );
  }

  stepsApplied.push(
    `Removed ${duplicatesRemoved} exact duplicate record(s) and ${invalidRowsRemoved} invalid/out-of-bounds vehicle record(s).`
  );
  stepsApplied.push(
    'Deferred median imputation (numerical) and mode/one-hot encoding (categorical) until AFTER the 80:20 train-test split to prevent data leakage.'
  );

  // Build column profiles
  const columnProfiles: ColumnProfile[] = headers.map((h) => {
    const mappedEntry = Object.entries(colMap).find(([, rawCol]) => rawCol === h);
    const mappedRole = mappedEntry ? mappedEntry[0] : 'unmapped_ignored';
    const isTarget = mappedRole === 'selling_price';
    const isNum = activeNumericFeatures.includes(mappedRole);
    return {
      name: h,
      mappedRole,
      inferredType: isTarget ? 'target' : isNum ? 'numerical' : 'categorical',
      missingCount: missingCounts[h] ?? 0,
      uniqueCount: uniqueValueSets[h]?.size ?? 0,
      sampleValues: sampleValueMap[h] ?? [],
      medianOrMode: sampleValueMap[h]?.[0] ?? 'N/A',
    };
  });

  const invalidReasons = Object.entries(invalidReasonMap).map(([reason, count]) => ({
    reason,
    count,
  }));

  return {
    cleanedRows: interimRows,
    report: {
      rawRows: rows.length,
      rawColumns: headers.length,
      cleanedRows: interimRows.length,
      duplicatesRemoved,
      invalidRowsRemoved,
      invalidReasons,
      missingValuesTotal: totalMissing,
      imputedCount: totalMissing,
      columnProfiles,
      stepsApplied,
    },
    activeNumericFeatures,
    activeCategoricalFeatures,
  };
}

/**
 * Fits median numerical imputers and top-K one-hot categorical encoders ONLY on training rows.
 */
export function fitPreprocessorOnTrain(
  trainRows: CleanedCarRow[],
  allCleanRows: CleanedCarRow[],
  activeNumericFeatures: string[],
  activeCategoricalFeatures: string[]
): PreprocessorState {
  const numericMedians: Record<string, number> = {};
  for (const col of activeNumericFeatures) {
    const vals = trainRows
      .map((r) => r[col])
      .filter((v): v is number => typeof v === 'number' && Number.isFinite(v))
      .sort((a, b) => a - b);
    numericMedians[col] = vals.length > 0 ? vals[Math.floor(vals.length / 2)] : 0;
  }

  const categoricalModes: Record<string, string> = {};
  const categoricalVocab: Record<string, string[]> = {};

  for (const col of activeCategoricalFeatures) {
    const freq = new Map<string, number>();
    for (const r of trainRows) {
      const val = r[col];
      if (typeof val === 'string' && val.trim()) {
        freq.set(val.trim(), (freq.get(val.trim()) ?? 0) + 1);
      }
    }
    const sorted = Array.from(freq.entries()).sort((a, b) => b[1] - a[1]);
    categoricalModes[col] = sorted[0]?.[0] ?? 'Other';

    // Keep up to 35 brands directly splittable and top 20 high-cardinality models + 'Other'
    const maxCategories = col === 'brand' ? 35 : col === 'model' ? 20 : 20;
    const topCats = sorted.slice(0, maxCategories).map(([k]) => k);
    if (sorted.length > maxCategories && !topCats.includes('Other')) {
      topCats.push('Other');
    }
    categoricalVocab[col] = topCats;
  }

  const featureNames: string[] = [...activeNumericFeatures];
  for (const col of activeCategoricalFeatures) {
    for (const cat of categoricalVocab[col]) {
      featureNames.push(`${col}=${cat}`);
    }
  }

  // Build dynamic UI input schema reflecting all valid categories from the dataset
  const featureSchema: FeatureFieldSchema[] = [];

  // Brand -> Models hierarchy if both exist
  const hasBrandModelHierarchy =
    activeCategoricalFeatures.includes('brand') && activeCategoricalFeatures.includes('model');
  const brandModelsMap: Record<string, string[]> = {};
  if (hasBrandModelHierarchy) {
    for (const r of allCleanRows) {
      const b = typeof r.brand === 'string' ? r.brand.trim() : '';
      const m = typeof r.model === 'string' ? r.model.trim() : '';
      if (b && m) {
        if (!brandModelsMap[b]) brandModelsMap[b] = [];
        if (!brandModelsMap[b].includes(m)) brandModelsMap[b].push(m);
      }
    }
    for (const b of Object.keys(brandModelsMap)) {
      brandModelsMap[b].sort();
    }
  }

  for (const col of activeCategoricalFeatures) {
    const allOptionsSet = new Set<string>();
    for (const r of allCleanRows) {
      const v = r[col];
      if (typeof v === 'string' && v.trim()) allOptionsSet.add(v.trim());
    }
    const options = Array.from(allOptionsSet).sort();
    featureSchema.push({
      key: col,
      label: CATEGORICAL_FEATURE_META[col]?.label ?? col,
      kind: 'categorical',
      options,
      defaultValue: categoricalModes[col] ?? options[0] ?? '',
      brandModelsMap: col === 'model' && hasBrandModelHierarchy ? brandModelsMap : undefined,
    });
  }

  for (const col of activeNumericFeatures) {
    const meta = NUMERIC_FEATURE_META[col] ?? {
      label: col,
      unit: '',
      min: 0,
      max: 1000000,
      step: 1,
    };
    const vals = allCleanRows
      .map((r) => r[col])
      .filter((v): v is number => typeof v === 'number' && Number.isFinite(v));
    const dataMin = vals.length > 0 ? Math.min(...vals) : meta.min;
    const dataMax = vals.length > 0 ? Math.max(...vals) : meta.max;
    featureSchema.push({
      key: col,
      label: meta.label,
      kind: 'numerical',
      unit: meta.unit,
      min: Math.min(meta.min, Math.floor(dataMin)),
      max: Math.max(meta.max, Math.ceil(dataMax)),
      step: meta.step,
      defaultValue: Number((numericMedians[col] ?? 0).toFixed(col === 'mileage' ? 1 : 0)),
    });
  }

  return {
    activeNumericFeatures,
    activeCategoricalFeatures,
    numericMedians,
    categoricalModes,
    categoricalVocab,
    featureNames,
    featureSchema,
    hasBrandModelHierarchy,
  };
}

/**
 * Transforms cleaned rows (or a single validated prediction payload) into numeric feature vectors.
 * Unseen categories safely map to 'Other' if present, or an all-zero one-hot slice.
 */
export function transformRows(
  rows: Record<string, string | number | undefined>[],
  state: PreprocessorState
): number[][] {
  return rows.map((row) => {
    const vec: number[] = [];

    for (const col of state.activeNumericFeatures) {
      const rawVal = row[col];
      const numVal =
        typeof rawVal === 'number' && Number.isFinite(rawVal)
          ? rawVal
          : typeof rawVal === 'string'
          ? parseNumericWithUnit(rawVal)
          : null;
      vec.push(numVal !== null && Number.isFinite(numVal) ? numVal : state.numericMedians[col] ?? 0);
    }

    for (const col of state.activeCategoricalFeatures) {
      const vocab = state.categoricalVocab[col] ?? [];
      const rawCat = row[col];
      const cat =
        typeof rawCat === 'string' && rawCat.trim()
          ? rawCat.trim()
          : state.categoricalModes[col] ?? '';

      const effectiveCat = vocab.includes(cat) ? cat : vocab.includes('Other') ? 'Other' : '';
      for (const v of vocab) {
        vec.push(v === effectiveCat ? 1 : 0);
      }
    }

    return vec;
  });
}

/**
 * Validates interactive user prediction inputs against the trained schema and real-world bounds.
 */
export function validatePredictionInputs(
  payload: PredictionRequest,
  state: PreprocessorState
): { validated: Record<string, string | number>; errors: Record<string, string> } {
  const validated: Record<string, string | number> = {};
  const errors: Record<string, string> = {};
  const currentYear = new Date().getFullYear();

  for (const field of state.featureSchema) {
    const raw = payload[field.key];
    if (field.kind === 'categorical') {
      const str = typeof raw === 'string' ? raw.trim() : String(raw ?? '').trim();
      if (!str) {
        errors[field.key] = `Please select or enter ${field.label.toLowerCase()}.`;
      } else {
        validated[field.key] = str;
      }
    } else {
      const num = typeof raw === 'number' ? raw : Number(raw);
      if (raw === '' || raw === undefined || raw === null || !Number.isFinite(num)) {
        errors[field.key] = `Please enter a valid number for ${field.label.toLowerCase()}.`;
        continue;
      }
      if (field.key === 'year') {
        if (!Number.isInteger(num) || num < 1985 || num > currentYear) {
          errors[field.key] = `Manufacturing year must be an integer between 1985 and ${currentYear}.`;
          continue;
        }
      } else if (field.key === 'km_driven') {
        if (num < 0 || num > 1500000) {
          errors[field.key] = 'Kilometres driven cannot be negative or exceed 15,00,000 km.';
          continue;
        }
      } else if (field.key === 'engine') {
        if (num < 0 || num > 8000) {
          errors[field.key] = 'Engine capacity must be between 0 (EV) and 8,000 CC.';
          continue;
        }
      } else if (field.key === 'mileage') {
        if (num < 0 || num > 80) {
          errors[field.key] = 'Mileage must be between 0 and 80 kmpl.';
          continue;
        }
      } else if (field.key === 'max_power') {
        if (num <= 0 || num > 1200) {
          errors[field.key] = 'Maximum power must be between 1 and 1,200 bhp.';
          continue;
        }
      } else if (field.key === 'seats') {
        if (!Number.isInteger(num) || num < 2 || num > 12) {
          errors[field.key] = 'Number of seats must be an integer between 2 and 12.';
          continue;
        }
      } else if (num < 0) {
        errors[field.key] = `${field.label} cannot be negative.`;
        continue;
      }
      validated[field.key] = num;
    }
  }

  return { validated, errors };
}
