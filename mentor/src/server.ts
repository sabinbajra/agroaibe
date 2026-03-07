import { readFile } from 'node:fs/promises';
import { createServer } from 'node:http';

import {
    HttpApiBuilder,
    HttpApiError,
    HttpMiddleware,
    HttpServer,
} from '@effect/platform';
import { NodeHttpServer, NodeRuntime } from '@effect/platform-node';
import { Effect, Layer } from 'effect';

import { api } from './api';
import { DISEASE_LABELS, DiseaseModel } from './disease';
import { IS_TOMATO_LABELS, IsTomatoModel } from './is-tomato';
import { buildPredictions } from './predictions';

const healthLive = HttpApiBuilder.group(api, 'health', handlers =>
    handlers.handle('health', () =>
        Effect.succeed({
            status: 'ok' as const,
        }),
    ),
);

const predictLive = HttpApiBuilder.group(api, 'predict', handlers =>
    handlers.handle('predict', ({ payload }) =>
        Effect.gen(function* () {
            const model = yield* DiseaseModel;
            const isTomatoModel = yield* IsTomatoModel;
            const file = payload.files[0];
            if (file === undefined) {
                return yield* Effect.fail(
                    new HttpApiError.InternalServerError(),
                );
            }

            const image = yield* Effect.tryPromise({
                try: () => readFile(file.path),
                catch: () => new HttpApiError.InternalServerError(),
            });

            const tomatoResult = yield* isTomatoModel.predict({
                image,
                labels: IS_TOMATO_LABELS,
            });

            const result = yield* model.predict({
                image,
                labels: DISEASE_LABELS,
            });

            return {
                isTomato: tomatoResult.isTomato,
                confidence: tomatoResult.confidence,
                predictions: buildPredictions(result.probabilities),
            };
        }).pipe(
            Effect.tapError(Effect.logError),
            Effect.mapError(() => new HttpApiError.InternalServerError()),
        ),
    ),
);

const apiLive = HttpApiBuilder.api(api).pipe(
    Layer.provide(healthLive),
    Layer.provide(predictLive),
    Layer.provide(DiseaseModel.layer),
    Layer.provide(IsTomatoModel.layer),
);

const serverLive = HttpApiBuilder.serve(HttpMiddleware.logger).pipe(
    Layer.provide(apiLive),
    HttpServer.withLogAddress,
    Layer.provide(NodeHttpServer.layer(createServer, { port: 3124 })),
);

Layer.launch(serverLive).pipe(NodeRuntime.runMain);
