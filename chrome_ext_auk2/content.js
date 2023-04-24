// chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
//   if (request.action === "test") {
//     const inputPrice = parseInt(request.inputPrice);
//     console.log('Input price loaded');

//     // location.reload();
//     setTimeout(() => {
//       const inputElement = document.querySelector('input[type="number"]');
//       const buttonElement = document.querySelector('.auk-button.transition-base.size-md[type="button"]');

//       inputElement.value = inputPrice;
//       buttonElement.click();
//       console.log('Clicked button');

//       setTimeout(() => {
//         const maxBidElem = document.querySelector('.row.bid span b');
//         const maxBid = parseInt(maxBidElem.textContent);

//         if (maxBid === inputPrice) {
//           const confirmBidElem = document.querySelector('button[data-gtm="confirm-bid"].btn-primary.big.fluid');
//           confirmBidElem.click();
//         }
//       }, 200);
//     }, 1000);
//   }
// });

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => { 
  if (request.action === "test") {
    const inputPrice = parseInt(request.inputPrice);
    console.log('Input price loaded');
    
    function clickButton() { 
      const inputElement = document.querySelector('input[type="number"]');
      const buttonElement = document.querySelector('.auk-button.transition-base.size-md[type="button"]');
      
      inputElement.value = inputPrice;

      
      buttonElement.click();
      console.log('Clicked button1'); 
      
      //setTimeout(checkMaxBid, 1000);
    }
    
    function checkMaxBid() {
      const maxBidElem = document.querySelector('.row.bid span b');
      const maxBid = parseInt(maxBidElem.textContent);
      console.log('located and parsed price'); 

      if (maxBid === inputPrice) {
        confirmBid() 
       
      }
    }
    
    function confirmBid() {
      const confirmBidElem = document.querySelector('button[data-gtm="confirm-bid"].btn-primary.big.fluid');
      console.log('Located max'); 
      confirmBidElem.click();
      console.log('Clicked confirm'); 
    }
    
    setTimeout(clickButton, 1000);    
  }
});