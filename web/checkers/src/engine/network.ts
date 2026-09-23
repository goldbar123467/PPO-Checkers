/**
 * Dependency-free forward pass of `checkers.rl.networks.CheckersNetwork`.
 *
 * Architecture: a 3×3 convolution stem (8 → 64 channels) with GroupNorm(8) and ReLU, six
 * residual blocks of two 3×3 convolutions with GroupNorm, a policy head (1×1 conv → GroupNorm →
 * ReLU → linear 128 → 128 logits) and a value head (1×1 conv → GroupNorm → ReLU → linear 64 → 64 →
 * ReLU → linear 64 → 1 → tanh). Weights come from `scripts/export_browser_policy.py`.
 */
import { ACTION_COUNT, OBSERVATION_PLANES, OBSERVATION_SIZE } from "./encoding";

export const TRUNK_CHANNELS = 64;
export const TRUNK_GROUPS = 8;
export const RESIDUAL_BLOCKS = 6;
export const GROUP_NORM_EPS = 1e-5;
const CELLS = 64;

export interface TensorRecord {
  readonly name: string;
  readonly shape: readonly number[];
  readonly offset: number;
  readonly length: number;
}

export interface PolicyManifest {
  readonly schema: string;
  readonly parameterCount: number;
  readonly weightsSha256: string;
  readonly weightsSizeBytes: number;
  readonly tensors: readonly TensorRecord[];
}

export interface NetworkOutput {
  readonly logits: Float32Array;
  readonly value: number;
}

export const BROWSER_POLICY_SCHEMA = "CHECKERS_BROWSER_POLICY_1";

function convRecords(prefix: string, outChannels: number, inChannels: number, kernel: number) {
  return [
    [`${prefix}.weight`, [outChannels, inChannels, kernel, kernel]],
    [`${prefix}.bias`, [outChannels]],
  ] as const;
}

function affineRecords(prefix: string, shape: readonly number[], biasSize: number) {
  return [
    [`${prefix}.weight`, shape],
    [`${prefix}.bias`, [biasSize]],
  ] as const;
}

/** The exact `state_dict` layout the weight file must follow. */
export const EXPECTED_TENSORS: readonly (readonly [string, readonly number[]])[] = [
  ...convRecords("stem.0", TRUNK_CHANNELS, OBSERVATION_PLANES, 3),
  ...affineRecords("stem.1", [TRUNK_CHANNELS], TRUNK_CHANNELS),
  ...Array.from({ length: RESIDUAL_BLOCKS }, (_, index) => {
    const prefix = `residual_blocks.${index}`;
    return [
      ...convRecords(`${prefix}.conv1`, TRUNK_CHANNELS, TRUNK_CHANNELS, 3),
      ...affineRecords(`${prefix}.norm1`, [TRUNK_CHANNELS], TRUNK_CHANNELS),
      ...convRecords(`${prefix}.conv2`, TRUNK_CHANNELS, TRUNK_CHANNELS, 3),
      ...affineRecords(`${prefix}.norm2`, [TRUNK_CHANNELS], TRUNK_CHANNELS),
    ];
  }).flat(),
  ...convRecords("policy_head.conv", 2, TRUNK_CHANNELS, 1),
  ...affineRecords("policy_head.norm", [2], 2),
  ...affineRecords("policy_head.output", [ACTION_COUNT, ACTION_COUNT], ACTION_COUNT),
  ...convRecords("value_head.conv", 1, TRUNK_CHANNELS, 1),
  ...affineRecords("value_head.norm", [1], 1),
  ...affineRecords("value_head.hidden", [CELLS, CELLS], CELLS),
  ...affineRecords("value_head.output", [1, CELLS], 1),
];

function isLittleEndian(): boolean {
  return new Uint8Array(new Uint16Array([1]).buffer)[0] === 1;
}

function sameShape(a: readonly number[], b: readonly number[]): boolean {
  return a.length === b.length && a.every((value, index) => value === b[index]);
}

/** Unfold a padded 3×3 neighbourhood of every cell into rows of `columns` (im2col). */
function unfold3x3(input: Float64Array, channels: number, columns: Float64Array): void {
  columns.fill(0, 0, channels * 9 * CELLS);
  for (let channel = 0; channel < channels; channel += 1) {
    const inBase = channel * CELLS;
    for (let tap = 0; tap < 9; tap += 1) {
      const dy = Math.floor(tap / 3) - 1;
      const dx = (tap % 3) - 1;
      const rowBase = (channel * 9 + tap) * CELLS;
      for (let y = Math.max(0, -dy); y < Math.min(8, 8 - dy); y += 1) {
        for (let x = Math.max(0, -dx); x < Math.min(8, 8 - dx); x += 1) {
          columns[rowBase + y * 8 + x] = input[inBase + (y + dy) * 8 + x + dx];
        }
      }
    }
  }
}

/**
 * Multiply `weight` (`outChannels × depth`) by unfolded `columns` (`depth × 64`) into `output`,
 * four output channels at a time so each column value is loaded once per block.
 */
function multiplyColumns(
  columns: Float64Array,
  depth: number,
  weight: Float32Array,
  bias: Float32Array,
  outChannels: number,
  output: Float64Array,
): void {
  for (let co = 0; co < outChannels; co += 4) {
    const base0 = co * CELLS;
    const base1 = base0 + CELLS;
    const base2 = base1 + CELLS;
    const base3 = base2 + CELLS;
    output.fill(bias[co], base0, base1);
    output.fill(bias[co + 1], base1, base2);
    output.fill(bias[co + 2], base2, base3);
    output.fill(bias[co + 3], base3, base3 + CELLS);
    const weight0 = co * depth;
    const weight1 = weight0 + depth;
    const weight2 = weight1 + depth;
    const weight3 = weight2 + depth;
    for (let k = 0; k < depth; k += 1) {
      const w0 = weight[weight0 + k];
      const w1 = weight[weight1 + k];
      const w2 = weight[weight2 + k];
      const w3 = weight[weight3 + k];
      const columnBase = k * CELLS;
      for (let cell = 0; cell < CELLS; cell += 1) {
        const value = columns[columnBase + cell];
        output[base0 + cell] += w0 * value;
        output[base1 + cell] += w1 * value;
        output[base2 + cell] += w2 * value;
        output[base3 + cell] += w3 * value;
      }
    }
  }
}

/** 1×1 convolution for the small policy and value heads. */
function pointwise(
  input: Float64Array,
  inChannels: number,
  weight: Float32Array,
  bias: Float32Array,
  outChannels: number,
  output: Float64Array,
): void {
  for (let co = 0; co < outChannels; co += 1) {
    const outBase = co * CELLS;
    output.fill(bias[co], outBase, outBase + CELLS);
    for (let ci = 0; ci < inChannels; ci += 1) {
      const w = weight[co * inChannels + ci];
      const inBase = ci * CELLS;
      for (let cell = 0; cell < CELLS; cell += 1) output[outBase + cell] += w * input[inBase + cell];
    }
  }
}

/** GroupNorm with biased variance (PyTorch semantics), optionally followed by ReLU. */
function groupNorm(
  values: Float64Array,
  channels: number,
  groups: number,
  gamma: Float32Array,
  beta: Float32Array,
  relu: boolean,
): void {
  const channelsPerGroup = channels / groups;
  const groupSize = channelsPerGroup * CELLS;
  for (let group = 0; group < groups; group += 1) {
    const start = group * groupSize;
    let sum = 0;
    for (let index = start; index < start + groupSize; index += 1) sum += values[index];
    const mean = sum / groupSize;
    let squares = 0;
    for (let index = start; index < start + groupSize; index += 1) {
      const centered = values[index] - mean;
      squares += centered * centered;
    }
    const inverse = 1 / Math.sqrt(squares / groupSize + GROUP_NORM_EPS);
    for (let c = 0; c < channelsPerGroup; c += 1) {
      const channel = group * channelsPerGroup + c;
      const scale = gamma[channel] * inverse;
      const shift = beta[channel] - mean * scale;
      const base = channel * CELLS;
      for (let index = base; index < base + CELLS; index += 1) {
        const normalized = values[index] * scale + shift;
        values[index] = relu && normalized < 0 ? 0 : normalized;
      }
    }
  }
}

function linear(
  input: Float64Array,
  weight: Float32Array,
  bias: Float32Array,
  output: Float64Array,
  relu: boolean,
): void {
  const inputs = input.length;
  for (let row = 0; row < output.length; row += 1) {
    let total = bias[row];
    const base = row * inputs;
    for (let column = 0; column < inputs; column += 1) total += weight[base + column] * input[column];
    output[row] = relu && total < 0 ? 0 : total;
  }
}

export class CheckersNetwork {
  readonly parameterCount: number;
  private readonly weights: ReadonlyMap<string, Float32Array>;
  private readonly trunk = new Float64Array(TRUNK_CHANNELS * CELLS);
  private readonly scratch = new Float64Array(TRUNK_CHANNELS * CELLS);
  private readonly branch = new Float64Array(TRUNK_CHANNELS * CELLS);
  private readonly columns = new Float64Array(TRUNK_CHANNELS * 9 * CELLS);
  private readonly policyPlanes = new Float64Array(2 * CELLS);
  private readonly policyLogits = new Float64Array(ACTION_COUNT);
  private readonly valuePlane = new Float64Array(CELLS);
  private readonly valueHidden = new Float64Array(CELLS);
  private readonly valueOutput = new Float64Array(1);

  private constructor(weights: ReadonlyMap<string, Float32Array>, parameterCount: number) {
    this.weights = weights;
    this.parameterCount = parameterCount;
  }

  /** Validate the manifest layout and wrap the raw little-endian float32 weight bytes. */
  static fromBuffer(buffer: ArrayBuffer, manifest: PolicyManifest): CheckersNetwork {
    if (!isLittleEndian()) throw new Error("browser policy weights require a little-endian CPU");
    if (manifest.schema !== BROWSER_POLICY_SCHEMA) throw new Error("unsupported policy schema");
    if (buffer.byteLength !== manifest.weightsSizeBytes || buffer.byteLength % 4 !== 0) {
      throw new Error("policy weight file has an unexpected size");
    }
    const values = new Float32Array(buffer);
    if (manifest.tensors.length !== EXPECTED_TENSORS.length) {
      throw new Error("policy tensor layout does not match the network");
    }
    const weights = new Map<string, Float32Array>();
    let offset = 0;
    manifest.tensors.forEach((record, index) => {
      const [name, shape] = EXPECTED_TENSORS[index];
      const length = shape.reduce((product, size) => product * size, 1);
      if (
        record.name !== name ||
        !sameShape(record.shape, shape) ||
        record.offset !== offset ||
        record.length !== length
      ) {
        throw new Error(`policy tensor layout mismatch for ${name}`);
      }
      weights.set(name, values.subarray(offset, offset + length));
      offset += length;
    });
    if (offset !== values.length || offset !== manifest.parameterCount) {
      throw new Error("policy weight file does not match its parameter count");
    }
    for (let index = 0; index < values.length; index += 1) {
      if (!Number.isFinite(values[index])) throw new Error("policy weights must be finite");
    }
    return new CheckersNetwork(weights, offset);
  }

  private tensor(name: string): Float32Array {
    return this.weights.get(name) as Float32Array;
  }

  private convNorm(
    prefix: string,
    norm: string,
    input: Float64Array,
    inChannels: number,
    output: Float64Array,
    relu: boolean,
  ): void {
    unfold3x3(input, inChannels, this.columns);
    multiplyColumns(
      this.columns,
      inChannels * 9,
      this.tensor(`${prefix}.weight`),
      this.tensor(`${prefix}.bias`),
      TRUNK_CHANNELS,
      output,
    );
    groupNorm(
      output,
      TRUNK_CHANNELS,
      TRUNK_GROUPS,
      this.tensor(`${norm}.weight`),
      this.tensor(`${norm}.bias`),
      relu,
    );
  }

  /** Return unmasked logits and the side to move's value estimate in [-1, 1]. */
  forward(observation: Float32Array): NetworkOutput {
    if (observation.length !== OBSERVATION_SIZE) {
      throw new Error("observation must contain 8 × 8 × 8 values");
    }
    const input = Float64Array.from(observation);
    this.convNorm("stem.0", "stem.1", input, OBSERVATION_PLANES, this.trunk, true);

    for (let block = 0; block < RESIDUAL_BLOCKS; block += 1) {
      const prefix = `residual_blocks.${block}`;
      this.convNorm(`${prefix}.conv1`, `${prefix}.norm1`, this.trunk, TRUNK_CHANNELS, this.scratch, true);
      this.convNorm(`${prefix}.conv2`, `${prefix}.norm2`, this.scratch, TRUNK_CHANNELS, this.branch, false);
      for (let index = 0; index < this.trunk.length; index += 1) {
        const sum = this.branch[index] + this.trunk[index];
        this.trunk[index] = sum > 0 ? sum : 0;
      }
    }

    pointwise(
      this.trunk,
      TRUNK_CHANNELS,
      this.tensor("policy_head.conv.weight"),
      this.tensor("policy_head.conv.bias"),
      2,
      this.policyPlanes,
    );
    groupNorm(
      this.policyPlanes,
      2,
      1,
      this.tensor("policy_head.norm.weight"),
      this.tensor("policy_head.norm.bias"),
      true,
    );
    linear(
      this.policyPlanes,
      this.tensor("policy_head.output.weight"),
      this.tensor("policy_head.output.bias"),
      this.policyLogits,
      false,
    );

    pointwise(
      this.trunk,
      TRUNK_CHANNELS,
      this.tensor("value_head.conv.weight"),
      this.tensor("value_head.conv.bias"),
      1,
      this.valuePlane,
    );
    groupNorm(
      this.valuePlane,
      1,
      1,
      this.tensor("value_head.norm.weight"),
      this.tensor("value_head.norm.bias"),
      true,
    );
    linear(
      this.valuePlane,
      this.tensor("value_head.hidden.weight"),
      this.tensor("value_head.hidden.bias"),
      this.valueHidden,
      true,
    );
    linear(
      this.valueHidden,
      this.tensor("value_head.output.weight"),
      this.tensor("value_head.output.bias"),
      this.valueOutput,
      false,
    );

    return { logits: Float32Array.from(this.policyLogits), value: Math.tanh(this.valueOutput[0]) };
  }
}
