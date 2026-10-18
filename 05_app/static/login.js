document.getElementById("f").addEventListener("submit", async ev => {
  ev.preventDefault();
  const b = document.getElementById("b"), msg = document.getElementById("msg");
  b.disabled = true; msg.textContent = "";
  try {
    const r = await fetch("api/login", {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({usuario: document.getElementById("u").value.trim().toLowerCase(),
                            clave: document.getElementById("c").value})
    });
    if (r.ok) { location.href = "./"; return; }
    const j = await r.json().catch(() => ({}));
    msg.textContent = j.detail || "No se pudo ingresar.";
    document.getElementById("c").value = "";
  } catch (e) {
    msg.textContent = "Sin conexión con el servidor.";
  } finally {
    b.disabled = false;
  }
});
