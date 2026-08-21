import eslint from '@eslint/js';
import tseslint from 'typescript-eslint';

export default tseslint.config(
  {
    ignores: [
      'dist/',
      'docs/dist/',
      'node_modules/',
      '*.js',
      'third_party/decoder.js',
    ],
  },
  eslint.configs.recommended,
  ...tseslint.configs.recommended,
  {
    // Node test scripts, not browser code.
    files: ['test/**/*.mjs'],
    languageOptions: {
      globals: {
        console: 'readonly',
        process: 'readonly',
      },
    },
  },
  {
    rules: {
      '@typescript-eslint/no-unused-vars': [
        'warn',
        {
          argsIgnorePattern: '^_',
          varsIgnorePattern: '^_',
          caughtErrorsIgnorePattern: '^_',
        },
      ],
    },
  },
);
