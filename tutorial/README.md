# Cube.js Tutorial

## Sekilas Cube Playground

__Playground link__: http://cubeopt.southeastasia.cloudapp.azure.com:4000/

Ada 4 fitur utama, yaitu:
* `Playground` untuk "ngoprek" running query, tabel output, visualisasi
* `Data Model` berisi list tabel yang dapat digunakan dari bigquery yang sudah tersambung (ada pada `Tables`) & data model dari tabel yang digunakan (ada pada `Files`) 
* `Frontend Integrations` berisi informasi endpoint REST API, Websockets, dan GraphQL untuk integrasi frontend
* `Connect to BI`

## Running Query di Cube Playground

Buka `Playground`

Pilih Cube (tabel data) yang ingin di-run (misal "user_generated_optimization")

Pilih dimension (biru) dan measure (kuning) yang ingin dipanggil, lalu klik `Run Query`

Untuk filter data, bisa menggunakan fitur `Filters` (tepat di bawah tombol `Run Query`). Order setiap dimension & measure bisa ditambahkan menggunakan fitur `Order` (bawah kanan)


## Local Dashboard

Pada `Playground`, klik fitur `Code`

Pilih jenis visualisasi, untuk preview output bisa dilihat dengan fitur `Preview`

Klik `Source` (Pojok kiri bawah) untuk download zip source code yang telah di-generate oleh Cube & `Config` (Pojok kiri bawah) untuk download file .env

### Running React App

_Catatan: Tutorial didasarkan pada yang ada di Github_

```
cd folder
npm install
npm start
```

Lalu buka `http://localhost:5173/vizard/preview/react-app`
