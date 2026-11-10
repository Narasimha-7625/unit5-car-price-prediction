import type { CartHyperparameters, CartTreeNode, DecisionPathStep } from './types.ts';

/**
 * Genuine Classification and Regression Trees (CART) DecisionTreeRegressor
 * using Squared-Error (Variance Reduction) splitting and Minimal Cost-Complexity Pruning.
 */
export class CartDecisionTreeRegressor {
  private params: CartHyperparameters;
  private featureNames: string[];
  private root: CartTreeNode | null = null;
  private totalTrainSamples = 0;
  private nodeCounter = 0;

  constructor(params: CartHyperparameters, featureNames: string[]) {
    this.params = params;
    this.featureNames = featureNames;
  }

  public fit(X: number[][], y: number[]): CartTreeNode {
    this.totalTrainSamples = X.length;
    this.nodeCounter = 0;

    const indices = Array.from({ length: X.length }, (_, i) => i);
    this.root = this.buildTree(X, y, indices, 0);

    if (this.params.ccp_alpha > 0 && this.root) {
      this.pruneCostComplexity(this.root, this.params.ccp_alpha);
    }

    return this.root;
  }

  public getTree(): CartTreeNode {
    if (!this.root) {
      throw new Error('CART model has not been fitted yet.');
    }
    return this.root;
  }

  public computeFeatureImportances(): number[] {
    if (!this.root || this.totalTrainSamples === 0) {
      return this.featureNames.map(() => 0);
    }

    const importances = new Array<number>(this.featureNames.length).fill(0);

    const accumulate = (node: CartTreeNode | null) => {
      if (!node || node.isLeaf || node.featureIndex === null) return;
      importances[node.featureIndex] += node.impurityDecrease;
      accumulate(node.left);
      accumulate(node.right);
    };

    accumulate(this.root);

    const sum = importances.reduce((acc, v) => acc + v, 0);
    if (sum <= 0) {
      return importances;
    }
    return importances.map((v) => v / sum);
  }

  private buildTree(
    X: number[][],
    y: number[],
    indices: number[],
    depth: number
  ): CartTreeNode {
    const n = indices.length;
    const nodeId = this.nodeCounter++;

    let sumY = 0;
    let sumSqY = 0;
    for (let i = 0; i < n; i++) {
      const target = y[indices[i]];
      sumY += target;
      sumSqY += target * target;
    }
    const meanY = sumY / n;
    const sse = Math.max(0, sumSqY - sumY * meanY);
    const mse = sse / n;

    // Stopping criteria matching Scikit-learn DecisionTreeRegressor
    const cannotSplit =
      depth >= this.params.max_depth ||
      n < this.params.min_samples_split ||
      n < 2 * this.params.min_samples_leaf ||
      mse <= 1e-9;

    if (cannotSplit) {
      return {
        nodeId,
        depth,
        isLeaf: true,
        samples: n,
        squaredError: mse,
        prediction: meanY,
        featureIndex: null,
        featureName: null,
        threshold: null,
        impurityDecrease: 0,
        left: null,
        right: null,
      };
    }

    const numFeatures = this.featureNames.length;
    let bestFeature = -1;
    let bestThreshold = 0;
    let bestImpurityReduction = 0;

    // Evaluate all candidate features and midpoint thresholds using variance reduction
    for (let f = 0; f < numFeatures; f++) {
      // Fast O(N) check for binary one-hot features (only 0 and 1 values)
      let isBinary = true;
      let count0 = 0;
      let sum0 = 0;
      let sq0 = 0;
      for (let i = 0; i < n; i++) {
        const idx = indices[i];
        const v = X[idx][f];
        if (v === 0) {
          count0++;
          const t = y[idx];
          sum0 += t;
          sq0 += t * t;
        } else if (v !== 1) {
          isBinary = false;
          break;
        }
      }

      if (isBinary) {
        const count1 = n - count0;
        if (
          count0 >= this.params.min_samples_leaf &&
          count1 >= this.params.min_samples_leaf
        ) {
          const sum1 = sumY - sum0;
          const sq1 = sumSqY - sq0;
          const leftSse = Math.max(0, sq0 - (sum0 * sum0) / count0);
          const rightSse = Math.max(0, sq1 - (sum1 * sum1) / count1);
          const weightedDecrease = (sse - (leftSse + rightSse)) / this.totalTrainSamples;
          if (weightedDecrease > bestImpurityReduction + 1e-9) {
            bestImpurityReduction = weightedDecrease;
            bestFeature = f;
            bestThreshold = 0.5;
          }
        }
        continue;
      }

      // General continuous feature scan: sort indices by feature value
      const sortedIdx = indices.slice().sort((a, b) => X[a][f] - X[b][f]);

      let leftSum = 0;
      let leftSqSum = 0;
      let rightSum = sumY;
      let rightSqSum = sumSqY;

      for (let splitPos = 0; splitPos < n - 1; splitPos++) {
        const curIdx = sortedIdx[splitPos];
        const nextIdx = sortedIdx[splitPos + 1];
        const curTarget = y[curIdx];
        const curVal = X[curIdx][f];
        const nextVal = X[nextIdx][f];

        leftSum += curTarget;
        leftSqSum += curTarget * curTarget;
        rightSum -= curTarget;
        rightSqSum -= curTarget * curTarget;

        const leftCount = splitPos + 1;
        const rightCount = n - leftCount;

        // Skip ties (identical feature values cannot be separated)
        if (Math.abs(nextVal - curVal) <= 1e-12) {
          continue;
        }

        if (
          leftCount < this.params.min_samples_leaf ||
          rightCount < this.params.min_samples_leaf
        ) {
          continue;
        }

        const leftSse = Math.max(0, leftSqSum - (leftSum * leftSum) / leftCount);
        const rightSse = Math.max(0, rightSqSum - (rightSum * rightSum) / rightCount);
        const sseReduction = sse - (leftSse + rightSse);
        const weightedDecrease = sseReduction / this.totalTrainSamples;

        if (weightedDecrease > bestImpurityReduction + 1e-9) {
          bestImpurityReduction = weightedDecrease;
          bestFeature = f;
          bestThreshold = (curVal + nextVal) / 2;
        }
      }
    }

    if (bestFeature === -1) {
      return {
        nodeId,
        depth,
        isLeaf: true,
        samples: n,
        squaredError: mse,
        prediction: meanY,
        featureIndex: null,
        featureName: null,
        threshold: null,
        impurityDecrease: 0,
        left: null,
        right: null,
      };
    }

    const bestLeftIndices: number[] = [];
    const bestRightIndices: number[] = [];
    for (let i = 0; i < n; i++) {
      const idx = indices[i];
      if (X[idx][bestFeature] <= bestThreshold) {
        bestLeftIndices.push(idx);
      } else {
        bestRightIndices.push(idx);
      }
    }

    if (bestLeftIndices.length === 0 || bestRightIndices.length === 0) {
      return {
        nodeId,
        depth,
        isLeaf: true,
        samples: n,
        squaredError: mse,
        prediction: meanY,
        featureIndex: null,
        featureName: null,
        threshold: null,
        impurityDecrease: 0,
        left: null,
        right: null,
      };
    }

    const leftNode = this.buildTree(X, y, bestLeftIndices, depth + 1);
    const rightNode = this.buildTree(X, y, bestRightIndices, depth + 1);

    return {
      nodeId,
      depth,
      isLeaf: false,
      samples: n,
      squaredError: mse,
      prediction: meanY,
      featureIndex: bestFeature,
      featureName: this.featureNames[bestFeature],
      threshold: bestThreshold,
      impurityDecrease: bestImpurityReduction,
      left: leftNode,
      right: rightNode,
    };
  }

  /**
   * Minimal Cost-Complexity Pruning (Breiman et al. CART):
   * R_alpha(T) = R(T) + alpha * |T|
   * Weakest-link pruning collapses any subtree whose effective alpha <= ccp_alpha.
   */
  private pruneCostComplexity(root: CartTreeNode, ccpAlpha: number): void {
    let prunedAny = true;
    while (prunedAny) {
      prunedAny = false;
      let weakestNode: CartTreeNode | null = null;
      let minEffectiveAlpha = Infinity;

      const postOrderInspect = (
        node: CartTreeNode
      ): { leafCount: number; subtreeWeightedR: number } => {
        const nodeWeightedR = (node.samples / this.totalTrainSamples) * node.squaredError;
        if (node.isLeaf || !node.left || !node.right) {
          return { leafCount: 1, subtreeWeightedR: nodeWeightedR };
        }
        const leftStats = postOrderInspect(node.left);
        const rightStats = postOrderInspect(node.right);
        const leafCount = leftStats.leafCount + rightStats.leafCount;
        const subtreeWeightedR = leftStats.subtreeWeightedR + rightStats.subtreeWeightedR;

        if (leafCount > 1) {
          const effectiveAlpha = (nodeWeightedR - subtreeWeightedR) / (leafCount - 1);
          if (effectiveAlpha < minEffectiveAlpha) {
            minEffectiveAlpha = effectiveAlpha;
            weakestNode = node;
          }
        }

        return { leafCount, subtreeWeightedR };
      };

      postOrderInspect(root);

      if (weakestNode !== null && minEffectiveAlpha <= ccpAlpha) {
        const target = weakestNode as CartTreeNode;
        target.isLeaf = true;
        target.featureIndex = null;
        target.featureName = null;
        target.threshold = null;
        target.impurityDecrease = 0;
        target.left = null;
        target.right = null;
        prunedAny = true;
      }
    }
  }
}

export function predictTreeRow(
  tree: CartTreeNode,
  row: number[]
): {
  prediction: number;
  path: DecisionPathStep[];
  leafSamples: number;
  leafSquaredError: number;
} {
  const path: DecisionPathStep[] = [];
  let current: CartTreeNode = tree;

  while (!current.isLeaf && current.featureIndex !== null && current.threshold !== null) {
    const val = row[current.featureIndex] ?? 0;
    const goLeft = val <= current.threshold;
    const featureLabel = current.featureName ?? `feature_${current.featureIndex}`;
    const isOneHot = featureLabel.includes('=');

    let ruleText = '';
    if (isOneHot) {
      const [col, cat] = featureLabel.split('=');
      ruleText = goLeft ? `${col} is NOT "${cat}"` : `${col} is "${cat}"`;
    } else {
      ruleText = goLeft
        ? `${featureLabel} <= ${formatThreshold(current.threshold)}`
        : `${featureLabel} > ${formatThreshold(current.threshold)}`;
    }

    path.push({
      nodeId: current.nodeId,
      depth: current.depth,
      featureName: featureLabel,
      threshold: current.threshold,
      observedValue: val,
      direction: goLeft ? 'left' : 'right',
      ruleText,
      samples: current.samples,
      nodeMeanPrice: current.prediction,
    });

    const next = goLeft ? current.left : current.right;
    if (!next) break;
    current = next;
  }

  return {
    prediction: current.prediction,
    path,
    leafSamples: current.samples,
    leafSquaredError: current.squaredError,
  };
}

export function computeTreeStats(tree: CartTreeNode): {
  totalNodes: number;
  leafNodes: number;
  actualMaxDepth: number;
} {
  let totalNodes = 0;
  let leafNodes = 0;
  let actualMaxDepth = 0;

  const walk = (node: CartTreeNode | null) => {
    if (!node) return;
    totalNodes++;
    if (node.depth > actualMaxDepth) actualMaxDepth = node.depth;
    if (node.isLeaf) {
      leafNodes++;
    } else {
      walk(node.left);
      walk(node.right);
    }
  };

  walk(tree);
  return { totalNodes, leafNodes, actualMaxDepth };
}

function formatThreshold(val: number): string {
  if (Math.abs(val) >= 1000) return Math.round(val).toLocaleString('en-IN');
  return Number(val.toFixed(2)).toString();
}
