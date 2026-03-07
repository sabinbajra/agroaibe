import { layer } from '@effect/vitest';
import { Effect } from 'effect';
import { describe, expect, it } from 'vitest';

import {
    argmax,
    IS_TOMATO_LABELS,
    IsTomatoLabel,
    IsTomatoModel,
    isTomatoFromLabel,
    softmax,
} from './is-tomato';

describe('is-tomato label mapping', () => {
    it('returns true for tomato label', () => {
        expect(IsTomatoLabel.Tomato).toBe('Tomato');
    });

    it('returns false for non-tomato label', () => {
        expect(IsTomatoLabel.NotTomato).toBe('Not_Tomato');
    });

    it('isTomatoFromLabel returns true for Tomato', () => {
        expect(isTomatoFromLabel(IsTomatoLabel.Tomato)).toBe(true);
    });

    it('isTomatoFromLabel returns false for NotTomato', () => {
        expect(isTomatoFromLabel(IsTomatoLabel.NotTomato)).toBe(false);
    });

    it('isTomatoFromLabel returns false for undefined', () => {
        expect(isTomatoFromLabel(undefined)).toBe(false);
    });
});

describe('softmax', () => {
    it('output sums to 1', () => {
        const result = softmax([1, 2, 3]);
        const sum = result.reduce((a, b) => a + b, 0);
        expect(sum).toBeCloseTo(1);
    });

    it('highest logit gets highest probability', () => {
        const result = softmax([0, 10, 0]);
        expect(argmax(result)).toBe(1);
    });
});

describe('IS_TOMATO_LABELS class order', () => {
    it('index 0 is Not_Tomato, index 1 is Tomato', () => {
        expect(IS_TOMATO_LABELS[0]).toBe(IsTomatoLabel.NotTomato);
        expect(IS_TOMATO_LABELS[1]).toBe(IsTomatoLabel.Tomato);
    });

    it('predictedIndex -> label -> isTomato mapping is correct for Tomato class', () => {
        const tomatoIndex = 1;
        const label = IS_TOMATO_LABELS[tomatoIndex];
        expect(isTomatoFromLabel(label)).toBe(true);
    });

    it('predictedIndex -> label -> isTomato mapping is correct for Not_Tomato class', () => {
        const notTomatoIndex = 0;
        const label = IS_TOMATO_LABELS[notTomatoIndex];
        expect(isTomatoFromLabel(label)).toBe(false);
    });
});

layer(IsTomatoModel.layer)('IsTomatoModel live inference', it => {
    it.effect(
        'classifies a known tomato leaf image as isTomato=true',
        () =>
            Effect.gen(function* () {
                const model = yield* IsTomatoModel;

                const response = yield* Effect.tryPromise({
                    try: () =>
                        fetch(
                            'https://content.ces.ncsu.edu/media/images/IMG_0019.jpg',
                        ).then(r => r.arrayBuffer()),
                    catch: e => new Error(String(e)),
                });

                const result = yield* model.predict({
                    image: new Uint8Array(response),
                    labels: IS_TOMATO_LABELS,
                });

                console.log(
                    `[is-tomato:live] predictedLabel=${result.predictedLabel} isTomato=${result.isTomato} confidence=${(result.confidence * 100).toFixed(2)}%`,
                );

                expect(result.isTomato).toBe(true);
                expect(result.predictedLabel).toBe(IsTomatoLabel.Tomato);
            }),
        { timeout: 30_000 },
    );
});
