chrome.runtime.onMessage.addListener(function (request, sender, sendResponse) {
  if (request.action === "toggleSidebar") {
    toggleSidebar();
  }
});

function toggleSidebar() {
  let sidebar = document.getElementById("my_chrome_extension_sidebar");

  if (sidebar) {
    sidebar.parentNode.removeChild(sidebar);
  } else {
    createSidebar();
  }
}

function createSidebar() {
  let sidebar = document.createElement("div");
  sidebar.id = "my_chrome_extension_sidebar";
  sidebar.style.width = "20%";
  sidebar.style.position = "fixed";
  sidebar.style.top = "0";
  sidebar.style.right = "0";
  sidebar.style.height = "100%";
  sidebar.style.overflow = "auto";
  sidebar.style.backgroundColor = "lightgray";
  sidebar.style.zIndex = "9999";
  sidebar.innerHTML = `
    <!-- Search input and notes list go here -->
  `;
  document.body.appendChild(sidebar);
}