import { access, mkdir, readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';

import { expect, layer } from '@effect/vitest';
import { Effect } from 'effect';

import { DISEASE_LABELS, DiseaseLabel, DiseaseModel } from './disease';

type ParityCase = {
    url: string;
    expectedLabel: DiseaseLabel;
    actualLabel: DiseaseLabel;
};

const aiDlParityCases = [
    {
        url: 'https://content.ces.ncsu.edu/media/images/IMG_0019.jpg',
        expectedLabel: DiseaseLabel.LateBlight,
        actualLabel: DiseaseLabel.LateBlight,
    },
    {
        url: 'https://content.ces.ncsu.edu/media/images/Shoemaker_7068.JPG',
        expectedLabel: DiseaseLabel.EarlyBlight,
        actualLabel: DiseaseLabel.EarlyBlight,
    },
    {
        url: 'https://content.ces.ncsu.edu/media/images/IMG_0679_1lAlIYx.jpeg',
        expectedLabel: DiseaseLabel.SeptoriaLeafSpot,
        actualLabel: DiseaseLabel.SeptoriaLeafSpot,
    },
    {
        url: 'https://content.ces.ncsu.edu/media/images/IMG_0675_NLNaTrA.jpeg',
        expectedLabel: DiseaseLabel.BacterialSpot,
        actualLabel: DiseaseLabel.SeptoriaLeafSpot,
    },
    {
        url: 'https://content.ces.ncsu.edu/media/images/K_Johnson_7082.JPG',
        expectedLabel: DiseaseLabel.BacterialSpot,
        actualLabel: DiseaseLabel.BacterialSpot,
    },
    {
        url: 'https://content.ces.ncsu.edu/media/images/spot_3.jpeg',
        expectedLabel: DiseaseLabel.SeptoriaLeafSpot,
        actualLabel: DiseaseLabel.BacterialSpot,
    },
    {
        url: 'https://content.ces.ncsu.edu/media/images/IMG_1770.jpeg',
        expectedLabel: DiseaseLabel.BacterialSpot,
        actualLabel: DiseaseLabel.BacterialSpot,
    },
    {
        url: 'https://vegcropshotline.org//wp-content/uploads/2016/06/IMG_1261.jpg',
        expectedLabel: DiseaseLabel.LeafMold,
        actualLabel: DiseaseLabel.LeafMold,
    },
    {
        url: 'https://extension.umn.edu/sites/extension.umn.edu/files/tomato-leaf-mold.jpg',
        expectedLabel: DiseaseLabel.LeafMold,
        actualLabel: DiseaseLabel.LeafMold,
    },
    {
        url: 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSlt7hrlrhJzmX1PvNH2kZOoE0ZxbEpGAKlKg&s',
        expectedLabel: DiseaseLabel.PowderyMildew,
        actualLabel: DiseaseLabel.PowderyMildew,
    },
    {
        url: 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcScdrqQKnBGqUicoA0NUmr61VQfec2aivNibw&s',
        expectedLabel: DiseaseLabel.PowderyMildew,
        actualLabel: DiseaseLabel.PowderyMildew,
    },
    {
        url: 'https://blogs.ifas.ufl.edu/stlucieco/files/2023/03/1-29.png',
        expectedLabel: DiseaseLabel.TomatoMosaicVirus,
        actualLabel: DiseaseLabel.TomatoMosaicVirus,
    },
    {
        url: 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcR7slzW94WJURpRu7Ezo0qOp5BM5j2pAmmuEA&s',
        expectedLabel: DiseaseLabel.SeptoriaLeafSpot,
        actualLabel: DiseaseLabel.TomatoMosaicVirus,
    },
    {
        url: 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSNghCWSXJ8KSYTYpXWraHGyvzxH2zVIiPXSQ&s',
        expectedLabel: DiseaseLabel.LateBlight,
        actualLabel: DiseaseLabel.SpiderMitesTwoSpottedSpiderMite,
    },
    {
        url: 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcROS1i3uSFLIsT7fx1V2qsXsqK7N7LnjnoOsQ&s',
        expectedLabel: DiseaseLabel.SeptoriaLeafSpot,
        actualLabel: DiseaseLabel.SpiderMitesTwoSpottedSpiderMite,
    },
] as const satisfies ReadonlyArray<{
    url: string;
    expectedLabel: DiseaseLabel;
    actualLabel: DiseaseLabel;
}>;

const imageCacheDir = path.join(process.cwd(), 'images');

const sanitizeUrlForCache = (url: string): string =>
    url.replace(/^https?:\/\//, '').replace(/[^a-zA-Z0-9._-]/g, '_');

const downloadImage = (url: string) =>
    Effect.tryPromise({
        try: async () => {
            const response = await fetch(url);
            if (!response.ok) {
                throw new Error(
                    `Failed to fetch image (${response.status} ${response.statusText}) from ${url}`,
                );
            }
            const buffer = await response.arrayBuffer();
            return new Uint8Array(buffer);
        },
        catch: error =>
            error instanceof Error
                ? error
                : new Error(`Failed to download image: ${String(error)}`),
    });

const getImageBytes = (parityCase: ParityCase) =>
    Effect.gen(function* () {
        const cacheFilePath = path.join(
            imageCacheDir,
            sanitizeUrlForCache(parityCase.url),
        );

        yield* Effect.promise(() => mkdir(imageCacheDir, { recursive: true }));

        const isCached = yield* Effect.promise(() =>
            access(cacheFilePath)
                .then(() => true)
                .catch(() => false),
        );

        if (isCached) {
            return yield* Effect.promise(() => readFile(cacheFilePath));
        }

        const downloaded = yield* downloadImage(parityCase.url);
        yield* Effect.promise(() => writeFile(cacheFilePath, downloaded));

        return downloaded;
    });

const formatRankedProbabilities = (
    probabilities: ReadonlyArray<number>,
): ReadonlyArray<{
    label: string;
    probability: number;
    rank: number;
}> =>
    probabilities
        .map((probability, index) => ({
            label: DISEASE_LABELS[index] ?? `class_${index}`,
            probability,
        }))
        .sort((a, b) => b.probability - a.probability)
        .map((entry, index) => ({
            ...entry,
            rank: index + 1,
        }));

const runParityCase = (parityCase: ParityCase) =>
    Effect.gen(function* () {
        const model = yield* DiseaseModel;
        const image = yield* getImageBytes(parityCase);
        const result = yield* model.predict({
            image,
            labels: DISEASE_LABELS,
        });

        const expected = parityCase.expectedLabel;
        const actual = parityCase.actualLabel;
        const ranked = formatRankedProbabilities(result.probabilities);
        const expectedEntry = ranked.find(entry => entry.label === expected);
        const rankedSummary = ranked
            .map(
                entry =>
                    `${entry.rank}.${entry.label}:${(entry.probability * 100).toFixed(2)}%`,
            )
            .join(' | ');

        console.log(
            `[disease-parity:strict] url=${parityCase.url} actual=${actual} expected=${expected} predicted=${result.predictedLabel} confidence=${(result.confidence * 100).toFixed(2)}% expectedRank=${expectedEntry?.rank ?? 'n/a'} all=${rankedSummary}`,
        );

        return {
            actual,
            expected,
            expectedEntry,
            result,
        };
    });

layer(DiseaseModel.layer)('parity', it => {
    it.effect.each(aiDlParityCases)(
        'strict $url -> $expectedLabel (actual: $actualLabel)',
        parityCase =>
            Effect.gen(function* () {
                const { expected, result } = yield* runParityCase(parityCase);
                expect(result.predictedLabel).toEqual(expected);
            }),
    );
});
