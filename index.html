<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Pure JS Client-Side Face Swap</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: system-ui, sans-serif; }
    body { background: #0f172a; color: #f8fafc; padding: 2rem; }
    .container { max-width: 900px; margin: 0 auto; background: #1e293b; padding: 1.5rem; border-radius: 12px; }
    h1 { margin-bottom: 1rem; text-align: center; }
    
    .upload-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-bottom: 1.5rem; }
    .box { border: 2px dashed #3b82f6; border-radius: 8px; padding: 1rem; text-align: center; background: #0f172a; }
    .box input { display: none; }
    .box label { cursor: pointer; display: block; font-weight: bold; padding: 0.5rem; }
    .preview { max-width: 100%; max-height: 200px; margin-top: 0.5rem; border-radius: 6px; display: none; }

    button { width: 100%; padding: 0.8rem; background: #2563eb; color: white; border: none; border-radius: 6px; font-size: 1rem; font-weight: bold; cursor: pointer; }
    button:hover { background: #1d4ed8; }

    .result-container { margin-top: 1.5rem; text-align: center; }
    canvas { max-width: 100%; border-radius: 8px; background: #000; margin-top: 1rem; }
  </style>
  
  <!-- Load TensorFlow.js and Face-API.js directly from CDN -->
  <script defer src="https://cdn.jsdelivr.net/npm/@vladmandic/face-api/dist/face-api.js"></script>
</head>
<body>

  <div class="container">
    <h1>Client-Side Face Swap</h1>

    <div class="upload-grid">
      <div class="box">
        <label for="faceInput">1. Upload Face Image</label>
        <input type="file" id="faceInput" accept="image/*" />
        <img id="faceImg" class="preview" alt="Face Preview" />
      </div>

      <div class="box">
        <label for="bodyInput">2. Upload Body/Clothing Image</label>
        <input type="file" id="bodyInput" accept="image/*" />
        <img id="bodyImg" class="preview" alt="Body Preview" />
      </div>
    </div>

    <button id="swapBtn">Swap Face onto Body</button>

    <div class="result-container">
      <h3>Result:</h3>
      <canvas id="outputCanvas"></canvas>
    </div>
  </div>

  <script>
    const faceInput = document.getElementById('faceInput');
    const bodyInput = document.getElementById('bodyInput');
    const faceImg = document.getElementById('faceImg');
    const bodyImg = document.getElementById('bodyImg');
    const swapBtn = document.getElementById('swapBtn');
    const canvas = document.getElementById('outputCanvas');
    const ctx = canvas.getContext('2d');

    // Handle local image previews
    function setupPreview(input, imgElement) {
      input.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
          imgElement.src = URL.createObjectURL(file);
          imgElement.style.display = 'block';
        }
      });
    }

    setupPreview(faceInput, faceImg);
    setupPreview(bodyInput, bodyImg);

    // Initialize models from CDN
    async function loadModels() {
      const MODEL_URL = 'https://cdn.jsdelivr.net/npm/@vladmandic/face-api/model/';
      await faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL);
      await faceapi.nets.faceLandmark64Net.loadFromUri(MODEL_URL);
    }

    loadModels();

    // Perform geometric face swap onto target canvas
    swapBtn.addEventListener('click', async () => {
      if (!faceImg.src || !bodyImg.src) {
        alert("Please upload both a face image and a body image first!");
        return;
      }

      swapBtn.innerText = "Processing...";

      // Detect face and landmarks on source (Face Image)
      const faceResult = await faceapi.detectSingleFace(faceImg, new faceapi.TinyFaceDetectorOptions()).withFaceLandmarks();
      // Detect face and landmarks on target (Body Image)
      const bodyResult = await faceapi.detectSingleFace(bodyImg, new faceapi.TinyFaceDetectorOptions()).withFaceLandmarks();

      if (!faceResult || !bodyResult) {
        alert("Could not detect clear faces in one or both images. Try clearer photos.");
        swapBtn.innerText = "Swap Face onto Body";
        return;
      }

      // Set canvas dimensions to target body image dimensions
      canvas.width = bodyImg.naturalWidth;
      canvas.height = bodyImg.naturalHeight;

      // 1. Draw target background (body image)
      ctx.drawImage(bodyImg, 0, 0);

      // 2. Extract bounding box and landmarks
      const srcBox = faceResult.detection.box;
      const tgtBox = bodyResult.detection.box;

      // 3. Create off-screen canvas to extract and clip source face
      const offCanvas = document.createElement('canvas');
      offCanvas.width = srcBox.width;
      offCanvas.height = srcBox.height;
      const offCtx = offCanvas.getContext('2d');

      // Draw source face portion to off-screen canvas
      offCtx.drawImage(
        faceImg, 
        srcBox.x, srcBox.y, srcBox.width, srcBox.height, 
        0, 0, srcBox.width, srcBox.height
      );

      // 4. Blend and warp onto target bounding area
      ctx.save();
      
      // Feathered oval mask for seamless blending
      ctx.beginPath();
      const centerX = tgtBox.x + tgtBox.width / 2;
      const centerY = tgtBox.y + tgtBox.height / 2;
      const radiusX = tgtBox.width / 2;
      const radiusY = tgtBox.height / 1.8;
      ctx.ellipse(centerX, centerY, radiusX, radiusY, 0, 0, 2 * Math.PI);
      ctx.clip();

      // Overlay swapped face
      ctx.drawImage(offCanvas, tgtBox.x, tgtBox.y, tgtBox.width, tgtBox.height);
      ctx.restore();

      swapBtn.innerText = "Swap Face onto Body";
    });
  </script>
</body>
</html>
