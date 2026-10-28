// The site is light only. A reader who picked dark here before (paradigm.xyz's preference,
// `paradigm.theme`) gets light too, from the first paint.
const root = document.documentElement;
root.dataset.theme = 'light';
root.style.colorScheme = 'light';
