'use strict';

// Progressive enhancement: content and native details remain readable without JS.
const steps = Array.from(document.querySelectorAll('.walk-step'));
const stepButtons = Array.from(document.querySelectorAll('[data-step]'));
const previous = document.getElementById('previous');
const next = document.getElementById('next');
let currentStep = 0;

function showStep(index) {
  currentStep = Math.max(0, Math.min(steps.length - 1, index));
  steps.forEach((step, i) => { step.hidden = i !== currentStep; });
  stepButtons.forEach((button, i) => {
    button.setAttribute('aria-controls', steps[i].id);
    if (i === currentStep) button.setAttribute('aria-current', 'step');
    else button.removeAttribute('aria-current');
  });
  previous.disabled = currentStep === 0;
  next.disabled = currentStep === steps.length - 1;
  document.getElementById('walk-status').textContent = `Bước ${currentStep + 1} / ${steps.length}`;
}

stepButtons.forEach(button => button.addEventListener('click', () => showStep(Number(button.dataset.step))));
previous.addEventListener('click', () => showStep(currentStep - 1));
next.addEventListener('click', () => showStep(currentStep + 1));
showStep(0);
document.querySelector('.walk-controls').hidden = false;
document.querySelector('.walk-footer').hidden = false;

const navigation = Array.from(document.querySelectorAll('.nav-links a'));
const sections = navigation.map(link => document.querySelector(link.getAttribute('href')));
let scrollPending = false;
function updateNavigation() {
  const headerHeight = document.querySelector('.site-header').getBoundingClientRect().height;
  document.documentElement.style.setProperty('--header', `${headerHeight}px`);
  let active = 0;
  sections.forEach((section, index) => {
    if (section.getBoundingClientRect().top <= headerHeight + 100) active = index;
  });
  navigation.forEach((link, index) => {
    if (index === active) link.setAttribute('aria-current', 'location');
    else link.removeAttribute('aria-current');
  });
  scrollPending = false;
}
function queueNavigationUpdate() {
  if (!scrollPending) {
    scrollPending = true;
    window.requestAnimationFrame(updateNavigation);
  }
}
window.addEventListener('scroll', queueNavigationUpdate, { passive: true });
window.addEventListener('resize', queueNavigationUpdate);
updateNavigation();
