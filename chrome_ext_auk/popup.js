document.getElementById('testButton').addEventListener('click', () => {
    const inputValue = document.getElementById('inputNumber').value;
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      chrome.tabs.sendMessage(tabs[0].id, { action: 'test', inputValue });
    });
  });
  
  document.getElementById('endButton').addEventListener('click', () => {
    const inputValue = document.getElementById('inputNumber').value;
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      chrome.tabs.sendMessage(tabs[0].id, { action: 'end', inputValue });
    });
  });