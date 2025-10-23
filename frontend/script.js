const sendBtn = document.getElementById("send");
const input = document.getElementById("message");
const chatBody = document.getElementById("chat-body");

let conversation = [];

// Tin nhắn mở đầu
window.onload = () => {
  const botMsg = document.createElement("div");
  botMsg.className = "message bot";
  botMsg.innerText = "Chào bạn! Hôm nay bạn cảm thấy thế nào? Cứ thoải mái chia sẻ với mình nhé.";
  chatBody.appendChild(botMsg);
  conversation.push({ role: "bot", content: botMsg.innerText });
};

// Gửi tin nhắn
sendBtn.addEventListener("click", async () => {
  const msg = input.value.trim();
  if (msg === "") return;

  const userMsg = document.createElement("div");
  userMsg.className = "message user";
  userMsg.innerText = msg;
  chatBody.appendChild(userMsg);
  input.value = "";
  conversation.push({ role: "user", content: msg });

  sendBtn.disabled = true;
  await sendToServer();
  sendBtn.disabled = false;
});

input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !sendBtn.disabled) sendBtn.click();
});

async function sendToServer() {
  const userMessages = conversation.filter(msg => msg.role === "user").map(msg => msg.content);
  const message = userMessages[userMessages.length - 1];
  const history = userMessages.length > 1 ? userMessages.slice(-3, -1).map(m => ({ content: m })) : [];

  try {
    const response = await fetch("http://127.0.0.1:8000/chatbot", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, history }),
    });

    if (!response.ok) throw new Error("Lỗi khi gọi backend");
    const data = await response.json();

    const botMsg = document.createElement("div");
    botMsg.className = "message bot";
    botMsg.innerText = data.content;
    chatBody.appendChild(botMsg);
    conversation.push({ role: "bot", content: data.content });
  } catch (error) {
    console.error(error);
    const botMsg = document.createElement("div");
    botMsg.className = "message bot";
    botMsg.innerText = "Có lỗi khi kết nối với máy chủ.";
    chatBody.appendChild(botMsg);
  }

  chatBody.scrollTop = chatBody.scrollHeight;
}
