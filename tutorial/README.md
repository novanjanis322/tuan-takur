# Integrasi Frontend dengan Cube.js API dan Query

Penjelasan langkah-langkah integrasikan Cube.js API dengan frontend & cara membuat query ke Cube.js API.

## Prasyarat

* Backend Cube.js yang telah dikonfigurasi.
* Proyek frontend yang menggunakan framework seperti React, Vue, atau Angular.
* API key Cube.js

## Instalasi Software Dev Kit Cube.js (React)

```bash
npm install @cubejs-client/core @cubejs-client/react
```

## Dependencies

```json
{
  "dependencies": {
    "@cubejs-client/core": "^0.35.0",
    "@cubejs-client/react": "^0.35.0",
    "@cubejs-client/ws-transport": "^0.35.23",
    "chart.js": "^4.4.2",
    "react": "^18.2.0",
    "react-chartjs-2": "^5.2.0",
    "react-dom": "^18.2.0"
  },
  "devDependencies": {
    "@types/react": "^18.2.56",
    "@types/react-dom": "^18.2.19",
    "@typescript-eslint/eslint-plugin": "^7.0.2",
    "@typescript-eslint/parser": "^7.0.2",
    "@vitejs/plugin-react": "^4.2.1",
    "eslint": "^8.56.0",
    "eslint-plugin-react-hooks": "^4.6.0",
    "eslint-plugin-react-refresh": "^0.4.5",
    "typescript": "^5.2.2",
    "vite": "^5.1.4"
  }
}
```

## Konfigurasi Cube.js Client

```tsx
const { apiUrl, apiToken, query1, pivotConfig1, chartType1, query2, pivotConfig2, chartType2, useWebSockets, useSubscription } = extractHashConfig(
  {
    // Konfigurasi API Cube
    apiUrl: import.meta.env.VITE_CUBE_API_URL || '',
    apiToken: import.meta.env.VITE_CUBE_API_TOKEN || '',

    // Konfigurasi Query & Pivot untuk Chart 1
    query1: JSON.parse(import.meta.env.VITE_CUBE_QUERY_1 || '{}') as Query,
    pivotConfig1: JSON.parse(
      import.meta.env.VITE_CUBE_PIVOT_CONFIG_1 || '{}'
    ) as PivotConfig,
    chartType1: import.meta.env.VITE_CHART_TYPE_1 as ChartType,

    // Konfigurasi Query & Pivot untuk Chart 2
    query2: JSON.parse(import.meta.env.VITE_CUBE_QUERY_2 || '{}') as Query,
    pivotConfig2: JSON.parse(
      import.meta.env.VITE_CUBE_PIVOT_CONFIG_2 || '{}'
    ) as PivotConfig,
    chartType2: import.meta.env.VITE_CHART_TYPE_2 as ChartType,

    // Pengaturan WebSocket dan Subscription
    // Mengkonversi string environment variables ke boolean
    useWebSockets: import.meta.env.VITE_CUBE_API_USE_WEBSOCKETS === 'true',
    useSubscription: import.meta.env.VITE_CUBE_API_USE_SUBSCRIPTION === 'true'
  });
```


