import { DISEASE_LABELS } from './disease';

export type Prediction = {
    readonly label: string;
    readonly confidence: number;
};

export const buildPredictions = (
    probabilities: ReadonlyArray<number>,
): ReadonlyArray<Prediction> =>
    probabilities
        .map((confidence, index) => ({
            label: DISEASE_LABELS[index] ?? `class_${index}`,
            confidence,
        }))
        .sort((left, right) => right.confidence - left.confidence)
        .slice(0, 3)
        .map(prediction => ({
            ...prediction,
            confidence: prediction.confidence,
        }));
