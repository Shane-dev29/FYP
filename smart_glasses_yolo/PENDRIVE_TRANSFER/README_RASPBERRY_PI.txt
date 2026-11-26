========================================================================
SMART ASSISTIVE GLASSES - RASPBERRY PI 5 DEPLOYMENT PACKAGE
========================================================================

CONTENTS OF THIS PENDRIVE PACKAGE:
------------------------------------------------------------------------
1_ONNX_MODELS_FOR_RASPBERRY_PI/
   - yolo26n_finetuned.onnx            (Model 1: Indoor Obstacles & People)
   - yolo26n_currency.onnx             (Model 2: Indian Currency Banknotes)
   - ch_PP-OCRv4_det_infer.onnx        (Model 3: Document Text Detector)
   - ch_PP-OCRv4_rec_infer.onnx        (Model 3: Document Text Recognizer)
   - ch_ppocr_mobile_v2.0_cls_infer.onnx(Model 3: Text Direction Classifier)

2_PYTORCH_MODELS_BACKUP/
   - yolo26n_finetuned.pt              (PyTorch backup weights)
   - yolo26n_currency.pt               (PyTorch backup weights)

3_PI_RUNNER_SCRIPTS/
   - raspberry_pi_inference.py         (Pi Edge Obstacle Detector)
   - currency_reader.py                (Pi Edge Banknote Reader)
   - document_reader.py                (Pi Edge Document Snap & Voice Reader)
   - data.yaml / currency_data.yaml    (Class ontologies)


STEP-BY-STEP SETUP ON RASPBERRY PI 5:
------------------------------------------------------------------------
1. Open Terminal on the Raspberry Pi.

2. Install lightweight edge dependencies (NO PyTorch needed!):
   pip install onnxruntime opencv-python numpy pyttsx3 rapidocr-onnxruntime

3. Copy this entire folder to your Pi home directory:
   cp -r /media/*/* /home/pi/smart_glasses_models/
   cd /home/pi/smart_glasses_models/3_PI_RUNNER_SCRIPTS/

4. RUN MODEL 1 - INDOOR OBSTACLE DETECTOR (Real-Time 41.5 FPS):
   python raspberry_pi_inference.py --model ../1_ONNX_MODELS_FOR_RASPBERRY_PI/yolo26n_finetuned.onnx --imgsz 416

5. RUN MODEL 2 - DEDICATED CURRENCY BANKNOTE READER:
   python currency_reader.py --model ../1_ONNX_MODELS_FOR_RASPBERRY_PI/yolo26n_currency.onnx --conf 0.30

6. RUN MODEL 3 - LIVE DOCUMENT & TEXT VOICE READER:
   python document_reader.py --engine rapidocr

========================================================================
All models are optimized with ARM NEON vectorization and run 100% offline!
========================================================================
