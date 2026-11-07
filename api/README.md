
### Lab 8 – Model Serving, API Testing & Performance Benchmarking

## 📌 Project Overview

This project focuses on deploying a deep learning model for **Industrial Manufacturing Defects Inspection** using FastAPI.

The system uses a trained ResNet-18 model from Week 6 to detect defects in casting product images. It also provides text analysis for industrial inspection notes, including PII masking and defect-related keyword identification.

The main goal is to make the trained AI model accessible through REST API endpoints and evaluate its performance.

## 🧠 Model & Dataset Information

| Component | Details |
|---|---|
| Model | ResNet-18 |
| Framework | PyTorch |
| Dataset | Casting Product Quality Dataset |
| Total Images | 6,633 |
| Defective Class | `def_front` |
| Non-defective Class | `ok_front` |
| Image Mode | Grayscale |

## ⚙️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming |
| PyTorch | Model inference |
| FastAPI | REST API development |
| Uvicorn | API server |
| Pytest | Automated testing |
| HTTPX | API requests and benchmarking |
| Pillow | Image processing |

## 🌐 API Endpoints

| Method | Endpoint | Function |
|---|---|---|
| GET | `/health` | Checks API and model status |
| POST | `/predict` | Predicts defects from casting images |
| POST | `/analyze` | Analyzes industrial inspection notes |

API documentation is available at:

`http://127.0.0.1:8000/docs`

## 🧪 API Testing Results

Automated testing was performed using Pytest.

**Command:**
```bash
pytest -q test_api.py
```

| Test Case | Purpose |
|---|---|
| Health Check | Verify API availability |
| Prediction Schema | Validate prediction response |
| Model Parity | Compare API and direct model predictions |
| Wrong File Type | Test file validation |
| File Too Large | Test upload size validation |
| Corrupt Image | Test invalid image handling |
| Missing File | Test missing upload handling |
| PII Masking | Verify personal information protection |

### Final Test Results

| Result | Count |
|---|---:|
| Total Tests | 8 |
| Passed | 8 |
| Failed | 0 |

**Status: All 8 API tests passed successfully!**

## 📊 Performance Benchmark

The API was benchmarked using 50 image prediction requests.

### Actual Benchmark Output

```text
n=50 p50=56 ms p95=70 ms max=341 ms
Throughput with 1 worker ~ 15.9 requests/s
```

### Performance Summary

| Metric | Result |
|---|---:|
| Total Requests | 50 |
| p50 Latency | 56 ms |
| p95 Latency | 70 ms |
| Maximum Latency | 341 ms |
| Workers | 1 |
| Throughput | 15.9 requests/second |

The results provide a performance baseline for the local industrial defect-inspection API.

## 🔐 Data Protection

The API includes:
- Image file validation.
- File size validation.
- Handling of corrupt images.
- PII masking in industrial inspection notes.
- Optional API-key authentication.

## 💡 Key Learnings

- Deploying a trained PyTorch model using FastAPI.
- Building REST API endpoints for image and text analysis.
- Testing API functionality using Pytest.
- Handling invalid image uploads and missing files.
- Protecting personal information through PII masking.
- Measuring API latency and throughput.
- Running and testing an API locally using Uvicorn.

## 📝 Conclusion & Reflection

This lab helped me understand how to convert a trained machine learning model into a working API application.

I successfully integrated my Week 6 ResNet-18 casting defect-detection model with FastAPI, implemented image prediction and industrial text-analysis endpoints, and completed all 8 automated API tests successfully.

The performance benchmark processed 50 requests with a median latency of 56 ms and throughput of approximately 15.9 requests per second using one worker.

This project strengthened my practical understanding of AI model deployment, API development, automated testing, and performance evaluation for industrial manufacturing applications.

## 🚀 Future Improvements

- Improve defect-detection accuracy.
- Test with more real-world casting images.
- Add API monitoring and logging.
- Evaluate performance with multiple workers.
- Explore cloud deployment for industrial applications.

---

**Project Domain:** Industrial Manufacturing Defects Inspection

**Model:** ResNet-18 | **API:** FastAPI | **Testing:** Pytest

`Python` `PyTorch` `FastAPI` `Uvicorn` `Pytest` `Industrial AI` `Computer Vision` `Model Deployment`

