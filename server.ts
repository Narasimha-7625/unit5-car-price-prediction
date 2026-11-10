import express from 'express';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createServer as createViteServer } from 'vite';
import {
  predictFromArtifact,
  trainCartPipelineFromCsv,
} from './src/ml/trainingPipeline.ts';
import type { CartModelArtifact } from './src/ml/types.ts';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const DATA_DIR = path.join(__dirname, 'data');
const MODELS_DIR = path.join(__dirname, 'models');
const SAMPLE_CSV_PATH = path.join(DATA_DIR, 'sample_indian_cars.csv');
const UPLOADED_PROJECT_CSV_PATH = path.join(
  __dirname,
  'public',
  'assets',
  'aistudio',
  'car_price_238cca5c.csv'
);
const SAVED_MODEL_PATH = path.join(MODELS_DIR, 'cart_model.json');
const PYTHON_DIR = path.join(__dirname, 'python_streamlit');

let cachedModel: CartModelArtifact | null = null;

async function ensureModelsDir(): Promise<void> {
  await fs.mkdir(MODELS_DIR, { recursive: true });
}

async function saveModelArtifact(artifact: CartModelArtifact): Promise<void> {
  await ensureModelsDir();
  await fs.writeFile(SAVED_MODEL_PATH, JSON.stringify(artifact), 'utf-8');
  cachedModel = artifact;
}

async function loadOrInitializeModel(): Promise<CartModelArtifact> {
  if (cachedModel) return cachedModel;

  try {
    const rawJson = await fs.readFile(SAVED_MODEL_PATH, 'utf-8');
    const parsed = JSON.parse(rawJson) as CartModelArtifact;
    if (parsed && parsed.tree && parsed.metrics && parsed.preprocessorState) {
      cachedModel = parsed;
      return parsed;
    }
  } catch {
    // Model artifact not yet persisted; initialize from project dataset
  }

  try {
    const uploadedCsv = await fs.readFile(UPLOADED_PROJECT_CSV_PATH, 'utf-8');
    const artifact = trainCartPipelineFromCsv(
      uploadedCsv,
      'car_price_238cca5c.csv (Uploaded Indian Used-Car Dataset)',
      'uploaded_csv'
    );
    await saveModelArtifact(artifact);
    return artifact;
  } catch {
    const csvText = await fs.readFile(SAMPLE_CSV_PATH, 'utf-8');
    const artifact = trainCartPipelineFromCsv(
      csvText,
      'Sample Indian Used Cars (Demonstration Dataset)',
      'sample_demo'
    );
    await saveModelArtifact(artifact);
    return artifact;
  }
}

async function readPythonDeliverableFiles(): Promise<
  { path: string; language: string; content: string }[]
> {
  const relativeFiles = [
    { rel: 'app.py', lang: 'python' },
    { rel: 'src/data_preprocessing.py', lang: 'python' },
    { rel: 'src/model_training.py', lang: 'python' },
    { rel: 'src/prediction.py', lang: 'python' },
    { rel: 'src/evaluation.py', lang: 'python' },
    { rel: 'src/visualization.py', lang: 'python' },
    { rel: 'requirements.txt', lang: 'text' },
    { rel: 'README.md', lang: 'markdown' },
  ];

  const results: { path: string; language: string; content: string }[] = [];
  for (const item of relativeFiles) {
    try {
      const full = path.join(PYTHON_DIR, item.rel);
      const content = await fs.readFile(full, 'utf-8');
      results.push({ path: item.rel, language: item.lang, content });
    } catch {
      // File may not exist yet during early boot
    }
  }
  return results;
}

async function startServer() {
  const app = express();
  app.use(express.json({ limit: '15mb' }));

  // Pre-load or build initial saved CART model before accepting requests
  await loadOrInitializeModel();

  app.get('/api/model', async (_req, res) => {
    try {
      const model = await loadOrInitializeModel();
      res.json(model);
    } catch {
      res.status(500).json({
        error: 'Unable to load the trained CART model. Please retrain using a valid dataset.',
      });
    }
  });

  app.post('/api/predict', async (req, res) => {
    try {
      const model = await loadOrInitializeModel();
      const outcome = predictFromArtifact(model, req.body ?? {});
      if (!outcome.ok) {
        res.status(400).json({
          error: 'Validation failed for one or more vehicle specifications.',
          fieldErrors: outcome.errors,
        });
        return;
      }
      res.json(outcome.result);
    } catch {
      res.status(500).json({
        error: 'An unexpected error occurred while evaluating the CART decision tree.',
      });
    }
  });

  app.post('/api/upload-retrain', async (req, res) => {
    try {
      const { csvText, fileName } = req.body ?? {};
      if (typeof csvText !== 'string' || !csvText.trim()) {
        res.status(400).json({ error: 'Please select a non-empty CSV file.' });
        return;
      }

      const safeName =
        typeof fileName === 'string' && fileName.trim()
          ? path.basename(fileName.trim())
          : 'uploaded_dataset.csv';

      const artifact = trainCartPipelineFromCsv(csvText, safeName, 'uploaded_csv');
      await saveModelArtifact(artifact);
      res.json(artifact);
    } catch (err) {
      const message =
        err instanceof Error
          ? err.message
          : 'Dataset processing or CART training failed. Check your CSV format and column names.';
      res.status(400).json({ error: message });
    }
  });

  app.post('/api/restore-sample', async (_req, res) => {
    try {
      const csvText = await fs.readFile(SAMPLE_CSV_PATH, 'utf-8');
      const artifact = trainCartPipelineFromCsv(
        csvText,
        'Sample Indian Used Cars (Demonstration Dataset)',
        'sample_demo'
      );
      await saveModelArtifact(artifact);
      res.json(artifact);
    } catch {
      res.status(500).json({
        error: 'Unable to restore the sample demonstration dataset.',
      });
    }
  });

  app.get('/api/sample-template', async (_req, res) => {
    try {
      const csvText = await fs.readFile(SAMPLE_CSV_PATH, 'utf-8');
      res.setHeader('Content-Type', 'text/csv; charset=utf-8');
      res.setHeader(
        'Content-Disposition',
        'attachment; filename="indian_used_cars_template.csv"'
      );
      res.send(csvText);
    } catch {
      res.status(404).json({ error: 'Sample template dataset is unavailable.' });
    }
  });

  app.get('/api/python-files', async (_req, res) => {
    const files = await readPythonDeliverableFiles();
    res.json({ files });
  });

  if (process.env.NODE_ENV === 'production') {
    const distDir = path.join(__dirname, 'dist');
    app.use(express.static(distDir));
    app.get('*', (_req, res) => {
      res.sendFile(path.join(distDir, 'index.html'));
    });
  } else {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  }

  const PORT = 3000;
  app.listen(PORT, '0.0.0.0', () => {
    console.log(`CART Used Car Valuation Server listening on port ${PORT}`);
  });
}

startServer().catch((err) => {
  console.error('Failed to start server:', err);
  process.exit(1);
});
