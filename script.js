function switchMode(mode) {
  document.getElementById("imageBtn").classList.toggle("active", mode === "image");
  document.getElementById("manualBtn").classList.toggle("active", mode === "manual");
  document.getElementById("imageSection").classList.toggle("active", mode === "image");
  document.getElementById("manualSection").classList.toggle("active", mode === "manual");
}

document.getElementById("damageImages").addEventListener("change", function (e) {
  const count = e.target.files.length;
  const info = document.getElementById("fileInfo");
  if (count > 0) {
    info.style.display = "block";
    info.innerHTML = `<strong>${count}</strong> image${count > 1 ? "s" : ""} selected`;
  } else {
    info.style.display = "none";
  }
});

document.querySelector(".btn-claim").addEventListener("click", async () => {
  const isImageMode = document.getElementById("imageSection").classList.contains("active");
  const formData = new FormData();

  if (isImageMode) {
    const files = document.getElementById("damageImages").files;
    if (files.length === 0) return alert("Please select at least one image");
    for (let file of files) formData.append("images", file);
  } else {
    const inputs = document.querySelectorAll("#manualSection [name]");
    let filled = false;
    inputs.forEach(input => {
      if (input.value.trim()) {
        formData.append(input.name, input.value.trim());
        filled = true;
      }
    });
    if (!filled) return alert("Please fill at least one field");
  }

  const btn = document.querySelector(".btn-claim");
  const origText = btn.innerText;
  btn.innerText = "Generating Professional Report...";
  btn.disabled = true;

  let resultBox = document.getElementById("resultBox");
  if (resultBox) resultBox.remove();

  try {
    const res = await fetch("/generate-claim", { 
      method: "POST",
      body: formData
    });

    const data = await res.json();
    
    const resultBox = document.createElement("div");
    resultBox.id = "resultBox";
    resultBox.className = "glass-container mt-5 p-5";
    resultBox.innerHTML = `
      <h3 style="color: #60a5fa; text-align: center; margin-bottom: 30px; font-weight: 800;">
        📄 Professional Claim Report Generated
      </h3>
      <pre style="white-space: pre-wrap; background: rgba(15,23,42,0.8); padding: 30px; border-radius: 18px; color: #e2e8f0; font-size: 1.15rem; line-height: 1.8; border: 1px solid rgba(96,165,250,0.3);">
${data.claim_report}
      </pre>
    `;
    document.querySelector(".content").appendChild(resultBox);
    resultBox.scrollIntoView({ behavior: "smooth" });
  } catch (err) {
    alert("Connection error. Is the server running?");
    console.error(err);
  } finally {
    btn.innerText = origText;
    btn.disabled = false;
  }
});