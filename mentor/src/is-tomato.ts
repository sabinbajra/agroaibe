import { Path } from '@effect/platform';
import { NodeContext } from '@effect/platform-node';
import AppRoot from 'app-root-path';
import { Context, Effect, Layer, Schema } from 'effect';
import * as ort from 'onnxruntime-node';
import sharp from 'sharp';

const getAppRoot = Effect.sync(() => AppRoot.toString());

export const IS_TOMATO_IMAGE_SIZE = 448;
export const IS_TOMATO_CHANNELS = 3;

export type NormalizationMode = 'none' | 'imagenet';

export const PREPROCESS = {
    imageSize: IS_TOMATO_IMAGE_SIZE,
    fit: 'cover' as const,
    rotateExif: false,
    kernel: sharp.kernel.lanczos3,
    normalization: 'none' as NormalizationMode,
};

export const IMAGENET_MEAN = [0.485, 0.456, 0.406] as const;
export const IMAGENET_STD = [0.229, 0.224, 0.225] as const;

export enum IsTomatoLabel {
    NotTomato = 'Not_Tomato',
    Tomato = 'Tomato',
}

export const IS_TOMATO_LABELS: ReadonlyArray<IsTomatoLabel> = [
    IsTomatoLabel.NotTomato,
    IsTomatoLabel.Tomato,
];

export interface IsTomatoPredictionInput {
    readonly image: Uint8Array | ArrayBuffer;
    readonly labels?: ReadonlyArray<IsTomatoLabel>;
}

export interface IsTomatoPredictionOutput {
    readonly predictedIndex: number;
    readonly predictedLabel: IsTomatoLabel | undefined;
    readonly isTomato: boolean;
    readonly confidence: number;
    readonly logits: ReadonlyArray<number>;
    readonly probabilities: ReadonlyArray<number>;
}

interface IIsTomatoModel {
    predict: (
        input: IsTomatoPredictionInput,
    ) => Effect.Effect<IsTomatoPredictionOutput, Error>;
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

export const isTomatoFromLabel = (
    label: IsTomatoLabel | undefined,
): boolean => label === IsTomatoLabel.Tomato;

const makeIsTomatoModel: Effect.Effect<IIsTomatoModel, never, Path.Path> =
    Effect.gen(function* () {
        const path = yield* Path.Path;
        const root = yield* getAppRoot;
        const filePath = path.join(
            root,
            'models',
            'is_tomato_RESNET34_HR_448x448_v1.onnx',
        );

        const session = yield* Effect.promise(() =>
            ort.InferenceSession.create(filePath),
        );

        const predict: IIsTomatoModel['predict'] = input =>
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
                    const rgb = await pipeline
                        .resize(PREPROCESS.imageSize, PREPROCESS.imageSize, {
                            fit: PREPROCESS.fit,
                            kernel: PREPROCESS.kernel,
                        })
                        .removeAlpha()
                        .raw()
                        .toBuffer();

                    const hw = PREPROCESS.imageSize * PREPROCESS.imageSize;
                    const tensorData = new Float32Array(
                        IS_TOMATO_CHANNELS * hw,
                    );

                    for (let i = 0; i < hw; i++) {
                        let r = rgb[i * 3]! / 255;
                        let g = rgb[i * 3 + 1]! / 255;
                        let b = rgb[i * 3 + 2]! / 255;

                        if (PREPROCESS.normalization === 'imagenet') {
                            r = (r - IMAGENET_MEAN[0]) / IMAGENET_STD[0];
                            g = (g - IMAGENET_MEAN[1]) / IMAGENET_STD[1];
                            b = (b - IMAGENET_MEAN[2]) / IMAGENET_STD[2];
                        }

                        tensorData[i] = r;
                        tensorData[hw + i] = g;
                        tensorData[2 * hw + i] = b;
                    }

                    const tensor = new ort.Tensor('float32', tensorData, [
                        1,
                        IS_TOMATO_CHANNELS,
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
                    const labels = input.labels ?? IS_TOMATO_LABELS;
                    const predictedLabel = labels[predictedIndex];

                    return {
                        predictedIndex,
                        predictedLabel,
                        isTomato: isTomatoFromLabel(predictedLabel),
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

export class IsTomatoModel extends Context.Tag('IsTomatoModel')<
    IsTomatoModel,
    IIsTomatoModel
>() {
    static layer = Layer.effect(IsTomatoModel, makeIsTomatoModel).pipe(
        Layer.provide(NodeContext.layer),
    );
}
