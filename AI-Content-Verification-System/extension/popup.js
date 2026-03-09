const BACKEND_URL = "http://127.0.0.1:8000";

const statusEl = document.getElementById("backend-status");
const toggleEl = document.getElementById("enableToggle");

// Check backend status
fetch(BACKEND_URL)
  .then(() => {
    statusEl.textContent = "Connected";
    statusEl.style.color = "green";
  })
  .catch(() => {
    statusEl.textContent = "Not running";
    statusEl.style.color = "red";
  });

// Load toggle state
chrome.storage.sync.get(["enabled"], (result) => {
  toggleEl.checked = result.enabled !== false;
});

// Save toggle state
toggleEl.addEventListener("change", () => {
  chrome.storage.sync.set({ enabled: toggleEl.checked });
});
