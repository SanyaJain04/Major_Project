chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.removeAll(() => {
    chrome.contextMenus.create({
      id: "scanImage",
      title: "Scan image with AI",
      contexts: ["image"]
    });

    chrome.contextMenus.create({
      id: "scanText",
      title: "Scan selected text with AI",
      contexts: ["selection"]
    });
  });
});

chrome.contextMenus.onClicked.addListener((info, tab) => {
  if (!tab?.id) return;

  if (info.menuItemId === "scanImage") {
    chrome.tabs.sendMessage(tab.id, {
      type: "IMAGE_SCAN",
      imageUrl: info.srcUrl
    }).catch(() => {
      console.warn("Content script not available on this page");
    });
  }

  if (info.menuItemId === "scanText") {
    chrome.tabs.sendMessage(tab.id, {
      type: "MANUAL_SCAN",
      text: info.selectionText
    }).catch(() => {
      console.warn("Content script not available on this page");
    });
  }
});
