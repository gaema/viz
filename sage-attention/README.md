# sage attention — low-precision scores

> **▶ [Open this demo](index.html)**

Keys and values are fixed. The query-scale slider changes the query. The score row is quantized to int8 against its own maximum and compared with exact attention. At the default scale the error is above zero. All-zero keys stay exact.
