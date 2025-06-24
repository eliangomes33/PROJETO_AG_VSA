// ====== CONSTANTES ======
const THEME_KEY = "app-theme";
const FONT_SIZE_KEY = "app-font-size";
const SOUNDS_ENABLED_KEY = "app-sounds-enabled";
const NOTIFICATIONS_ENABLED_KEY = "app-notifications-enabled";

// ====== FUNÇÕES DE ÁUDIO ======
const audio = {
    success: new Audio('assets/audio/success.mp3'), // Confirme o caminho e o nome do arquivo
    error: new Audio('assets/audio/error.mp3'),    // Confirme o caminho e o nome do arquivo
    // Adicione mais sons se desejar
};

function playSound(type) {
    if (localStorage.getItem(SOUNDS_ENABLED_KEY) === 'true' && audio[type]) {
        audio[type].play().catch(e => console.warn("Erro ao tocar som:", e));
    }
}

// ====== FUNÇÕES DE NOTIFICAÇÃO ======
function requestNotificationPermission() {
    if (!("Notification" in window)) {
        console.warn("Este navegador não suporta notificações de desktop.");
        return;
    }
    Notification.requestPermission().then(permission => {
        if (permission === "granted") {
            localStorage.setItem(NOTIFICATIONS_ENABLED_KEY, 'true');
            console.log("Permissão de notificação concedida.");
        } else {
            localStorage.setItem(NOTIFICATIONS_ENABLED_KEY, 'false');
            console.warn("Permissão de notificação negada.");
            alert("Permissão de notificação negada. Por favor, habilite nas configurações do seu navegador."); // Mensagem fixa em PT-BR
        }
    });
}

// showNotification agora espera title e message como strings diretas
function showNotification(title, message) {
    if (localStorage.getItem(NOTIFICATIONS_ENABLED_KEY) === 'true' && Notification.permission === "granted") {
        new Notification(title, { body: message, icon: 'assets/img/logo.png' });
    }
}

// ====== FUNÇÕES DE APLICAR ESTILOS/CONFIGURAÇÕES ======
function applyInitialTheme() {
    const savedTheme = localStorage.getItem(THEME_KEY) || 'light';
    document.documentElement.setAttribute('data-theme', savedTheme);
}

function applyInitialFontSize() {
    const savedFontSize = localStorage.getItem(FONT_SIZE_KEY) || '16';
    document.documentElement.style.fontSize = `${savedFontSize}px`;
}

function applyTheme(theme) {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem(THEME_KEY, theme);
}

function applyFontSize(size) {      
    document.documentElement.style.fontSize = `${size}px`;
    localStorage.setItem(FONT_SIZE_KEY, size);
}

// Carrega as configurações salvas e atualiza os controles (se existirem)
function loadSettings() {
    // Tema
    const savedTheme = localStorage.getItem(THEME_KEY) || "light";
    applyTheme(savedTheme);

    // Tamanho da fonte
    const savedFontSize = localStorage.getItem(FONT_SIZE_KEY) || "16";
    applyFontSize(savedFontSize);

    // Sons
    const soundsEnabled = localStorage.getItem(SOUNDS_ENABLED_KEY) === 'true';
    const enableSoundsCheckbox = document.getElementById("enable-sounds");
    if (enableSoundsCheckbox) enableSoundsCheckbox.checked = soundsEnabled;

    // Notificações no navegador
    const notificationsEnabled = localStorage.getItem(NOTIFICATIONS_ENABLED_KEY) === 'true';
    const enableNotificationsCheckbox = document.getElementById("enable-notifications");
    if (enableNotificationsCheckbox) enableNotificationsCheckbox.checked = notificationsEnabled;
}

// ====== INICIALIZAÇÃO ======
document.addEventListener("DOMContentLoaded", () => {
    loadSettings(); // Carrega e aplica todas as configurações ao carregar o DOM
});

// Se estiver na página de configurações, configura os listeners específicos
if (window.location.pathname.includes("configuracoes.html")) {
    document.getElementById("theme")?.addEventListener("change", (e) => {
        applyTheme(e.target.value);
    });

    document.getElementById("font-size")?.addEventListener("input", (e) => {
        const size = e.target.value;
        applyFontSize(size);
        document.getElementById("font-size-value").textContent = `${size}px`;
    });

    document.getElementById("enable-sounds")?.addEventListener("change", (e) => {
        localStorage.setItem(SOUNDS_ENABLED_KEY, e.target.checked);
        if (e.target.checked) playSound('success'); // Toca um som de teste ao habilitar
    });

    document.getElementById("test-sound-btn")?.addEventListener("click", () => {
        playSound('success'); // Toca som de teste
    });

    document.getElementById("enable-notifications")?.addEventListener("change", (e) => {
        localStorage.setItem(NOTIFICATIONS_ENABLED_KEY, e.target.checked);
        if (e.target.checked) {
            requestNotificationPermission(); // Pede permissão se habilitado
        }
    });

    document.getElementById("test-notification-btn")?.addEventListener("click", () => {
        // Mensagens de teste fixas em PT-BR
        showNotification("Teste de Notificação", "Esta é uma notificação de teste!");
    });

    document.getElementById("reset-btn")?.addEventListener("click", () => {
        if (confirm('Restaurar configurações padrão?')) { // Confirmação fixa em PT-BR
            localStorage.clear(); // Limpa todas as configurações salvas
            location.reload(); // Recarrega a página para aplicar os padrões
        }
    });
}