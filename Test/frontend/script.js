document.getElementById("predictForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  const formData = new FormData(e.target);
  const payload = Object.fromEntries(formData.entries());
  Object.keys(payload).forEach(k => payload[k] = parseFloat(payload[k]));

  try {
    const res = await fetch("http://127.0.0.1:8000/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    const data = await res.json();
    document.getElementById("result").innerHTML = `
      <h3>${data.prediction === 1 ? "💔 High Risk" : "❤️ Low Risk"}</h3>
      <p>Probability: ${(data.probability * 100).toFixed(2)}%</p>
      <p>${data.advice}</p>
    `;
  } catch (err) {
    document.getElementById("result").innerHTML = `<p style="color:red;">Error connecting to server</p>`;
  }
});
