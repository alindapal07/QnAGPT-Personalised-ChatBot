(function() {
    function detectAndApplyTheme() {
        const app = document.querySelector('.stApp') || document.body;
        let isDark = false;
        if (app) {
            const bg = window.getComputedStyle(app).backgroundColor;
            const m = bg.match(/\d+/g);
            if (m && m.length >= 3) {
                const lum = 0.299 * parseInt(m[0], 10) + 0.587 * parseInt(m[1], 10) + 0.114 * parseInt(m[2], 10);
                isDark = lum < 128;
            } else if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
                isDark = true;
            }
        } else if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
            isDark = true;
        }

        const themeStr = isDark ? 'dark' : 'light';
        document.documentElement.setAttribute('data-theme', themeStr);
        document.body.setAttribute('data-theme', themeStr);
        if (app) {
            app.setAttribute('data-theme', themeStr);
        }
    }

    detectAndApplyTheme();
    setTimeout(detectAndApplyTheme, 150);
    setTimeout(detectAndApplyTheme, 500);

    try {
        const observer = new MutationObserver(function() {
            detectAndApplyTheme();
        });
        const target = document.querySelector('.stApp') || document.body;
        if (target) {
            observer.observe(target, { attributes: true, attributeFilter: ['class', 'style'] });
        }
    } catch (e) {}

    if (window.matchMedia) {
        window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', detectAndApplyTheme);
    }
})();
