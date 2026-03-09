// ================================
// CONFIG
// ================================
const TEXT_BACKEND_URL = "http://127.0.0.1:8000/analyze/text";
const IMAGE_BACKEND_URL = "http://127.0.0.1:8000/analyze/image";

const CONFIDENCE_THRESHOLD = 0.75;
const IMAGE_CONFIDENCE_THRESHOLD = 0.75;
const SCAN_INTERVAL_MS = 3000;

// ================================
// UTILS
// ================================
function isValidText(text) {
  return text && text.length > 20 && text.length < 500;
}

function blurElement(element, confidence, label) {
  element.style.filter = "blur(5px)";
  element.style.transition = "filter 0.3s ease";
  element.title = `${label} detected (${(confidence * 100).toFixed(1)}%)`;
}

function blurImage(img, confidence, label) {
  img.style.filter = "blur(10px)";
  img.style.transition = "filter 0.3s ease";
  img.title = `${label} image detected (${(confidence * 100).toFixed(1)}%)`;
}

// ================================
// TEXT EXTRACTION
// ================================
function getTextNodes() {
  const walker = document.createTreeWalker(
    document.body,
    NodeFilter.SHOW_TEXT,
    {
      acceptNode(node) {
        if (!node.parentElement) return NodeFilter.FILTER_REJECT;

        const tag = node.parentElement.tagName;
        if (["SCRIPT", "STYLE", "NOSCRIPT"].includes(tag)) {
          return NodeFilter.FILTER_REJECT;
        }

        return NodeFilter.FILTER_ACCEPT;
      }
    }
  );

  const nodes = [];
  let current;

  while ((current = walker.nextNode())) {
    const text = current.textContent.trim();
    if (isValidText(text)) {
      nodes.push(current);
    }
  }
  return nodes;
}

// ================================
// BACKEND CALLS
// ================================
async function analyzeText(text) {
  try {
    const res = await fetch(TEXT_BACKEND_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text })
    });
    return await res.json();
  } catch (err) {
    console.error("Text backend error:", err);
    return null;
  }
}

async function analyzeImage(imgElement) {
  try {
    const response = await fetch(imgElement.src);
    const blob = await response.blob();

    const formData = new FormData();
    formData.append("file", blob, "image.jpg");

    const res = await fetch(IMAGE_BACKEND_URL, {
      method: "POST",
      body: formData
    });

    return await res.json();
  } catch (err) {
    console.error("Image scan failed:", err);
    return null;
  }
}

// ================================
// TEXT SCAN LOGIC
// ================================
async function scanText() {
  chrome.storage.sync.get(["enabled"], async (result) => {
    if (result.enabled === false) return;

    const textNodes = getTextNodes();

    for (const node of textNodes) {
      const el = node.parentElement;

      if (el.dataset.textScanned === "true") continue;

      const result = await analyzeText(node.textContent);

      if (
        result &&
        result.prediction !== "Neutral" &&
        result.confidence >= CONFIDENCE_THRESHOLD
      ) {
        blurElement(el, result.confidence, result.prediction);
      }

      el.dataset.textScanned = "true";
    }
  });
}

// ================================
// IMAGE SCAN LOGIC
// ================================
async function scanImages() {
  chrome.storage.sync.get(["enabled"], async (result) => {
    if (result.enabled === false) return;

    const images = document.querySelectorAll("img");

    for (const img of images) {
      if (img.dataset.imageScanned === "true") continue;
      if (!img.src || img.width < 100 || img.height < 100) continue;

      const result = await analyzeImage(img);

      if (
        result &&
        result.prediction === "Illicit" &&
        result.confidence >= IMAGE_CONFIDENCE_THRESHOLD
      ) {
        blurImage(img, result.confidence, result.prediction);
      }

      img.dataset.imageScanned = "true";
    }
  });
}

// ================================
// MAIN LOOP
// ================================
setInterval(() => {
  scanText();
  scanImages();
}, SCAN_INTERVAL_MS);

// ================================
// MANUAL SCAN (RIGHT CLICK / POPUP)
// ================================
chrome.runtime.onMessage.addListener(async (request) => {
  console.log("Message received:", request);

  if (request.type === "MANUAL_SCAN") {
    const result = await analyzeText(request.text);

    if (!result) {
      alert("Backend not reachable");
      return;
    }

    alert(
      `Prediction: ${result.prediction}\nConfidence: ${(result.confidence * 100).toFixed(1)}%`
    );
  }
  if (request.type === "IMAGE_SCAN") {
  const imageUrl = request.imageUrl;

  try {
    const response = await fetch(imageUrl);
    const blob = await response.blob();

    const formData = new FormData();
    formData.append("file", blob, "image.jpg");

    const res = await fetch("http://127.0.0.1:8000/analyze/image", {
      method: "POST",
      body: formData
    });

    const result = await res.json();

    alert(
      `Image Scan Result\n\nPrediction: ${result.prediction}\nConfidence: ${(result.confidence * 100).toFixed(1)}%`
    );

  } catch (err) {
    alert("Failed to scan image");
    console.error(err);
  }
}

});
