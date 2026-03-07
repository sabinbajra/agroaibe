## Quick Start

1. Download the model onnx files.

[tomato_resnet34_448x448.onnx](https://www.kaggle.com/models/naiarafernandez/tomato-diseases-resnet34-onnx) into `./mentor/models/tomato_resnet34_448x448.onnx`.
[is_tomato_RESNET34_HR_448x448_v1.onnx](https://www.kaggle.com/models/naiarafernandez/istomato-resnet34-v1-448x448)
into `./mentor/models/is_tomato_RESNET34_HR_448x448_v1.onnx`

2. Install dependencies.

```bash
cd mentor
# --force only required because of a version mismatch i don't want to fix right now
npm i --force
```

3. Run server.

```bash
npm start
```

## API

### `POST /predict`

Accepts a multipart form upload with at least one image file in the `files` field (the server uses the first file).

Example:

```bash
curl -sS -X POST -F "files=@./images/example.jpg" http://localhost:3124/predict | jq
```

Response:

```json
{
  "isTomato": true,
  "confidence": 0.98,
  "predictions": [
    { "label": "healthy", "probability": 0.91 }
  ]
}
```

- `isTomato`: whether the image is classified as a tomato plant.
- `confidence`: confidence for the tomato/non-tomato classification.
- `predictions`: disease probabilities for tomato classes.

If upload/model processing fails, the endpoint returns `500 Internal Server Error`.
