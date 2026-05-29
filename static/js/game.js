// static/js/game.js — shared utilities (non-page-specific)

// Smooth scroll for landing page dots
document.addEventListener('DOMContentLoaded', () => {
  const dots = document.querySelectorAll('.dot');
  dots.forEach((dot, i) => {
    dot.addEventListener('click', () => {
      dots.forEach(d => d.classList.remove('active'));
      dot.classList.add('active');
    });
  });
});
