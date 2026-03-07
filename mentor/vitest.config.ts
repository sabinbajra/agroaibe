import * as path from 'node:path';

import { defineConfig } from 'vitest/config';

// see https://docs.nestjs.com/recipes/swc#vitest
export default defineConfig({
    test: {
        setupFiles: [path.join(__dirname, 'setup-tests.ts')],
        include: ['src/**/*.{test,spec}.?(c|m)[jt]s?(x)'],
        reporters: ['dot'],
        fakeTimers: { toFake: undefined },
    },
});
