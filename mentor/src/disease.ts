import { access } from 'node:fs/promises';
import { Path } from '@effect/platform';
import { NodeContext } from '@effect/platform-node';
import AppRoot from 'app-root-path';
import { Context, Effect, Layer, Schema } from 'effect';
import * as ort from 'onnxruntime-node';
import sharp from 'sharp';

const getAppRoot = Effect.sync(() => AppRoot.toString());

export const MODEL_IMAGE_SIZE = 448;
export const MODEL_CHANNELS = 3;
export const MODEL_INPUT_LENGTH =
    MODEL_CHANNELS * MODEL_IMAGE_SIZE * MODEL_IMAGE_SIZE;
export type NormalizationMode = 'none' | 'imagenet';

export const PREPROCESS = {
    stageOneSize: 600,
    imageSize: 448,
    fit: 'fill' as const,
    rotateExif: false,
    kernel: sharp.kernel.linear,
    normalization: 'none' as NormalizationMode,
};

export const IMAGENET_MEAN = [0.485, 0.456, 0.406] as const;
export const IMAGENET_STD = [0.229, 0.224, 0.225] as const;

export enum DiseaseLabel {
    BacterialSpot = 'Bacterial_spot',
    EarlyBlight = 'Early_blight',
    LateBlight = 'Late_blight',
    LeafMold = 'Leaf_Mold',
    SeptoriaLeafSpot = 'Septoria_leaf_spot',
    SpiderMitesTwoSpottedSpiderMite = 'Spider_mites Two-spotted_spider_mite',
    TargetSpot = 'Target_Spot',
    TomatoYellowLeafCurlVirus = 'Tomato_Yellow_Leaf_Curl_Virus',
    TomatoMosaicVirus = 'Tomato_mosaic_virus',
    Healthy = 'healthy',
    PowderyMildew = 'powdery_mildew',
}

export const DISEASE_LABELS: ReadonlyArray<DiseaseLabel> = [
    DiseaseLabel.BacterialSpot,
    DiseaseLabel.EarlyBlight,
    DiseaseLabel.LateBlight,
    DiseaseLabel.LeafMold,
    DiseaseLabel.SeptoriaLeafSpot,
    DiseaseLabel.SpiderMitesTwoSpottedSpiderMite,
    DiseaseLabel.TargetSpot,
    DiseaseLabel.TomatoYellowLeafCurlVirus,
    DiseaseLabel.TomatoMosaicVirus,
    DiseaseLabel.Healthy,
    DiseaseLabel.PowderyMildew,
];

export interface DiseasePredictionInput {
    /**
     * Raw uploaded image bytes (e.g. from multipart upload).
     */
    readonly image: Uint8Array | ArrayBuffer;
    /**
     * Optional class labels aligned by output index.
     */
    readonly labels?: ReadonlyArray<DiseaseLabel>;
}

export interface DiseasePredictionOutput {
    readonly predictedIndex: number;
    readonly predictedLabel: string | undefined;
    readonly confidence: number;
    readonly logits: ReadonlyArray<number>;
    readonly probabilities: ReadonlyArray<number>;
}

interface IDiseaseModel {
    predict: (
        input: DiseasePredictionInput,
    ) => Effect.Effect<DiseasePredictionOutput, Error>;
}

const OnnxOutputSchema = Schema.Array(Schema.Number).pipe(
    Schema.filter(logits => logits.length > 0, {
        message: () => 'ONNX output is empty.',
    }),
);

const decodeOnnxOutput = Schema.decodeUnknownSync(OnnxOutputSchema);

export const softmax = (
    values: ReadonlyArray<number>,
): ReadonlyArray<number> => {
    const max = Math.max(...values);
    const exps = values.map(value => Math.exp(value - max));
    const sum = exps.reduce((acc, value) => acc + value, 0);

    return exps.map(value => value / sum);
};

export const argmax = (values: ReadonlyArray<number>): number =>
    values.reduce(
        (bestIndex, value, index) =>
            value > (values[bestIndex] ?? 0) ? index : bestIndex,
        0,
    );

const resolveModelPath = (
    path: Path.Path,
    root: string,
): Effect.Effect<string, Error> => {
    const candidate = path.join(root, 'models', 'tomato_resnet34_448x448.onnx');

    const checkPath = (candidate: string) =>
        Effect.tryPromise({
            try: () => access(candidate).then(() => candidate),
            catch: () =>
                new Error(`Disease ONNX model not found at: ${candidate}`),
        });

    const primary = checkPath(candidate);

    return primary.pipe(
        Effect.mapError(
            () =>
                new Error(`No disease ONNX model found. Checked: ${candidate}`),
        ),
    );
};

const interpolateBilinearAlignCorners = (
    source: Float32Array,
    channels: number,
    inHeight: number,
    inWidth: number,
    outHeight: number,
    outWidth: number,
): Float32Array => {
    const output = new Float32Array(channels * outHeight * outWidth);
    const inHw = inHeight * inWidth;
    const outHw = outHeight * outWidth;

    const yLow = new Int32Array(outHeight);
    const yHigh = new Int32Array(outHeight);
    const yLerp = new Float32Array(outHeight);
    const xLow = new Int32Array(outWidth);
    const xHigh = new Int32Array(outWidth);
    const xLerp = new Float32Array(outWidth);

    const yScale = outHeight > 1 ? (inHeight - 1) / (outHeight - 1) : 0;
    const xScale = outWidth > 1 ? (inWidth - 1) / (outWidth - 1) : 0;

    for (let y = 0; y < outHeight; y++) {
        const coord = y * yScale;
        const low = Math.floor(coord);
        const high = Math.ceil(coord);
        yLow[y] = low;
        yHigh[y] = high;
        yLerp[y] = coord - low;
    }

    for (let x = 0; x < outWidth; x++) {
        const coord = x * xScale;
        const low = Math.floor(coord);
        const high = Math.ceil(coord);
        xLow[x] = low;
        xHigh[x] = high;
        xLerp[x] = coord - low;
    }

    for (let channel = 0; channel < channels; channel++) {
        const channelOffsetIn = channel * inHw;
        const channelOffsetOut = channel * outHw;

        for (let y = 0; y < outHeight; y++) {
            const yl = yLow[y]!;
            const yh = yHigh[y]!;
            const yWeight = yLerp[y]!;

            for (let x = 0; x < outWidth; x++) {
                const xl = xLow[x]!;
                const xh = xHigh[x]!;
                const xWeight = xLerp[x]!;

                const v00 = source[channelOffsetIn + yl * inWidth + xl]!;
                const v01 = source[channelOffsetIn + yl * inWidth + xh]!;
                const v10 = source[channelOffsetIn + yh * inWidth + xl]!;
                const v11 = source[channelOffsetIn + yh * inWidth + xh]!;

                const top = v00 + (v01 - v00) * xWeight;
                const bottom = v10 + (v11 - v10) * xWeight;

                output[channelOffsetOut + y * outWidth + x] =
                    top + (bottom - top) * yWeight;
            }
        }
    }

    return output;
};

const interpolateBilinearHalfPixel = (
    source: Float32Array,
    channels: number,
    inHeight: number,
    inWidth: number,
    outHeight: number,
    outWidth: number,
): Float32Array => {
    const output = new Float32Array(channels * outHeight * outWidth);
    const inHw = inHeight * inWidth;
    const outHw = outHeight * outWidth;

    const yScale = inHeight / outHeight;
    const xScale = inWidth / outWidth;

    for (let channel = 0; channel < channels; channel++) {
        const channelOffsetIn = channel * inHw;
        const channelOffsetOut = channel * outHw;

        for (let y = 0; y < outHeight; y++) {
            const srcY = (y + 0.5) * yScale - 0.5;
            const y0 = Math.max(0, Math.min(inHeight - 1, Math.floor(srcY)));
            const y1 = Math.max(0, Math.min(inHeight - 1, y0 + 1));
            const yWeight = srcY - y0;

            for (let x = 0; x < outWidth; x++) {
                const srcX = (x + 0.5) * xScale - 0.5;
                const x0 = Math.max(0, Math.min(inWidth - 1, Math.floor(srcX)));
                const x1 = Math.max(0, Math.min(inWidth - 1, x0 + 1));
                const xWeight = srcX - x0;

                const v00 = source[channelOffsetIn + y0 * inWidth + x0]!;
                const v01 = source[channelOffsetIn + y0 * inWidth + x1]!;
                const v10 = source[channelOffsetIn + y1 * inWidth + x0]!;
                const v11 = source[channelOffsetIn + y1 * inWidth + x1]!;

                const top = v00 + (v01 - v00) * xWeight;
                const bottom = v10 + (v11 - v10) * xWeight;

                output[channelOffsetOut + y * outWidth + x] =
                    top + (bottom - top) * yWeight;
            }
        }
    }

    return output;
};

const makeDiseaseModel: Effect.Effect<IDiseaseModel, Error, Path.Path> =
    Effect.gen(function* () {
        const path = yield* Path.Path;

        const root = yield* getAppRoot;
        const filePath = yield* resolveModelPath(path, root);

        const session = yield* Effect.promise(() =>
            ort.InferenceSession.create(filePath),
        );

        const predict: IDiseaseModel['predict'] = input =>
            Effect.tryPromise({
                try: async () => {
                    const inputName = session.inputNames[0];
                    if (inputName === undefined) {
                        throw new Error('ONNX session has no input name.');
                    }

                    const data =
                        input.image instanceof Uint8Array
                            ? input.image
                            : new Uint8Array(input.image);

                    const pipeline = PREPROCESS.rotateExif
                        ? sharp(data).rotate()
                        : sharp(data);
                    const { data: rgbOriginal, info } = await pipeline
                        .toColourspace('srgb')
                        .removeAlpha()
                        .raw()
                        .toBuffer({ resolveWithObject: true });

                    if (info.channels !== MODEL_CHANNELS) {
                        throw new Error(
                            `Expected ${MODEL_CHANNELS} channels after RGB conversion, got ${info.channels}.`,
                        );
                    }

                    const originalHw = info.width * info.height;
                    const originalChw = new Float32Array(
                        MODEL_CHANNELS * originalHw,
                    );

                    for (let i = 0; i < originalHw; i++) {
                        originalChw[i] = rgbOriginal[i * 3]! / 255;
                        originalChw[originalHw + i] =
                            rgbOriginal[i * 3 + 1]! / 255;
                        originalChw[2 * originalHw + i] =
                            rgbOriginal[i * 3 + 2]! / 255;
                    }

                    const stageOneChw = interpolateBilinearHalfPixel(
                        originalChw,
                        MODEL_CHANNELS,
                        info.height,
                        info.width,
                        PREPROCESS.stageOneSize,
                        PREPROCESS.stageOneSize,
                    );

                    const hw = PREPROCESS.imageSize * PREPROCESS.imageSize;
                    const tensorData = interpolateBilinearAlignCorners(
                        stageOneChw,
                        MODEL_CHANNELS,
                        PREPROCESS.stageOneSize,
                        PREPROCESS.stageOneSize,
                        PREPROCESS.imageSize,
                        PREPROCESS.imageSize,
                    );

                    if (PREPROCESS.normalization === 'imagenet') {
                        for (let i = 0; i < hw; i++) {
                            tensorData[i] =
                                (tensorData[i]! - IMAGENET_MEAN[0]) /
                                IMAGENET_STD[0];
                            tensorData[hw + i] =
                                (tensorData[hw + i]! - IMAGENET_MEAN[1]) /
                                IMAGENET_STD[1];
                            tensorData[2 * hw + i] =
                                (tensorData[2 * hw + i]! - IMAGENET_MEAN[2]) /
                                IMAGENET_STD[2];
                        }
                    }

                    const tensor = new ort.Tensor('float32', tensorData, [
                        1,
                        MODEL_CHANNELS,
                        PREPROCESS.imageSize,
                        PREPROCESS.imageSize,
                    ]);

                    const results = await session.run({ [inputName]: tensor });
                    const outputName =
                        session.outputNames[0] ?? Object.keys(results)[0];
                    if (outputName === undefined) {
                        throw new Error('ONNX session has no output name.');
                    }
                    const output = results[outputName];
                    if (output === undefined) {
                        throw new Error(
                            `ONNX output "${outputName}" is missing.`,
                        );
                    }

                    const outputData = (output as { data: unknown }).data;
                    const logits = decodeOnnxOutput(
                        ArrayBuffer.isView(outputData)
                            ? Array.from(outputData as any)
                            : outputData,
                    );

                    const probabilities = softmax(logits);
                    const predictedIndex = argmax(probabilities);
                    const predictedLabel = input.labels?.[predictedIndex];

                    return {
                        predictedIndex,
                        predictedLabel,
                        confidence: probabilities[predictedIndex] ?? 0,
                        logits,
                        probabilities,
                    };
                },
                catch: error =>
                    error instanceof Error
                        ? error
                        : new Error(
                              `Model prediction failed: ${String(error)}`,
                          ),
            });

        return { predict };
    });

export class DiseaseModel extends Context.Tag('DiseaseModel')<
    DiseaseModel,
    IDiseaseModel
>() {
    static layer = Layer.effect(DiseaseModel, makeDiseaseModel).pipe(
        Layer.provide(NodeContext.layer),
    );
}
