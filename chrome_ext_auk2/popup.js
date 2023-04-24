document.addEventListener("DOMContentLoaded", () => {
  const inputPrice = document.getElementById("input_price");
  const testButton = document.getElementById("test");
  const endButton = document.getElementById("end");
  const statusArea = document.getElementById("status");

  testButton.onclick = () => {
    const price = inputPrice.value;
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      chrome.tabs.sendMessage(tabs[0].id, { action: "test", inputPrice: price });
    });
  };

  function updateTime() {
    const now = new Date();
    const formattedTime = `${now.getDate()}. ${now.getMonth() + 1}. ${now.getFullYear()}, ${now.getHours()}:${now.getMinutes()}:${now.getSeconds()}`;
    statusArea.value = formattedTime;
  }

  setInterval(updateTime, 1000);
});