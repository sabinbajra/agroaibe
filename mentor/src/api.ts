import {
    HttpApi,
    HttpApiEndpoint,
    HttpApiError,
    HttpApiGroup,
    HttpApiSchema,
    Multipart,
} from '@effect/platform';
import { Schema } from 'effect';

export const health = HttpApiEndpoint.get('health', '/health').addSuccess(
    Schema.Struct({
        status: Schema.Literal('ok'),
    }),
);

const PredictionSchema = Schema.Struct({
    label: Schema.String,
    confidence: Schema.Number,
});

export const predict = HttpApiEndpoint.post('predict', '/predict')
    .setPayload(
        HttpApiSchema.Multipart(
            Schema.Struct({
                files: Multipart.FilesSchema,
            }),
        ),
    )
    .addSuccess(
        Schema.Struct({
            isTomato: Schema.Boolean,
            confidence: Schema.Number,
            predictions: Schema.Array(PredictionSchema),
        }),
    )
    .addError(HttpApiError.InternalServerError);

export const api = HttpApi.make('agrovision').add(
    HttpApiGroup.make('health').add(health),
).add(HttpApiGroup.make('predict').add(predict));
