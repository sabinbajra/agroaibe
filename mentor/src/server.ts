import { readFile } from 'node:fs/promises';
import { createServer } from 'node:http';

import {
    HttpApiBuilder,
    HttpApiError,
    HttpMiddleware,
    HttpServer,
    HttpServerResponse,
    HttpServerRequest,
} from '@effect/platform';
import { NodeHttpServer, NodeRuntime } from '@effect/platform-node';
import { Effect, Layer, pipe } from 'effect';

import { api } from './api';
import { DISEASE_LABELS, DiseaseModel } from './disease';
import { IS_TOMATO_LABELS, IsTomatoModel } from './is-tomato';
import { buildPredictions } from './predictions';

// ============================================
// CORS MIDDLEWARE - Fixed for Effect platform
// ============================================
const corsMiddleware: HttpMiddleware.HttpMiddleware = (request) =>
    Effect.gen(function* () {
        // Handle preflight OPTIONS requests
        if (request.method === 'OPTIONS') {
            console.log('[CORS] Handling OPTIONS preflight request');
            return HttpServerResponse.empty({
                status: 204,
                headers: {
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
                    'Access-Control-Allow-Headers': 'Content-Type, Accept, Origin',
                    'Access-Control-Max-Age': '86400',
                },
            });
        }

        // For non-OPTIONS requests, continue to next middleware
        return yield* Effect.fail(null);
    });

const addCorsHeaders: HttpMiddleware.HttpMiddleware = (request) =>
    Effect.gen(function* () {
        // Process the request through the API
        const response = yield* HttpServer.request(request);

        // Add CORS headers to the response
        return HttpServerResponse.fromResponse(response, {
            headers: {
                ...response.headers,
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
                'Access-Control-Allow-Headers': 'Content-Type, Accept, Origin',
            },
        });
    });

// ============================================
// HEALTH ENDPOINT
// ============================================
const healthLive = HttpApiBuilder.group(api, 'health', handlers =>
    handlers.handle('health', () =>
        Effect.succeed({
            status: 'ok' as const,
        }),
    ),
);

// ============================================
// PREDICT ENDPOINT
// ============================================
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

// ============================================
// API LAYER
// ============================================
const apiLive = HttpApiBuilder.api(api).pipe(
    Layer.provide(healthLive),
    Layer.provide(predictLive),
    Layer.provide(DiseaseModel.layer),
    Layer.provide(IsTomatoModel.layer),
);

// ============================================
// SERVER CONFIGURATION WITH CORS
// ============================================
const serverLive = HttpApiBuilder.serve(
    HttpMiddleware.make((request) =>
        Effect.gen(function* () {
            // Handle OPTIONS preflight
            if (request.method === 'OPTIONS') {
                console.log('[CORS] Handling OPTIONS preflight');
                return HttpServerResponse.empty({
                    status: 204,
                    headers: {
                        'Access-Control-Allow-Origin': '*',
                        'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
                        'Access-Control-Allow-Headers': 'Content-Type, Accept, Origin',
                        'Access-Control-Max-Age': '86400',
                    },
                });
            }

            // Process normal request
            const response = yield* HttpServer.request(request);

            // Add CORS headers
            return HttpServerResponse.fromResponse(response, {
                headers: {
                    ...response.headers,
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
                    'Access-Control-Allow-Headers': 'Content-Type, Accept, Origin',
                },
            });
        })
    ),
    HttpMiddleware.logger,
).pipe(
    Layer.provide(apiLive),
    HttpServer.withLogAddress,
    Layer.provide(NodeHttpServer.layer(createServer, { port: 3124 })),
);

// ============================================
// START SERVER
// ============================================
Layer.launch(serverLive).pipe(NodeRuntime.runMain);
